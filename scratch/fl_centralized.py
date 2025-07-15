# Filename: run_federated_experiment.py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import collections
import argparse
import os
import time
import json
import pandas as pd
from fl_api import (
    get_default_args,
    load_and_distribute_data_api,
    create_model_api,
    client_update_api,
    aggregate_weights_fedavg_api,
    structurally_prune_model,
    FL_STATE
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Função auxiliar para carregar os clientes selecionados
def load_selected_clients(path="selected_clients.jsonl"):
    selected = {}
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            selected[entry["round"]] = entry["clients"]
    return selected

# Função auxiliar para medir o tamanho do modelo (em MB)
def get_model_size_mb(model):
    num_params = model.count_params()
    size_bytes = num_params * 4  # float32 = 4 bytes
    return size_bytes / (1024 ** 2)

clients_this_round = []
test_metrics_log = [] 

# Etapa 1: Configurar os argumentos
config = get_default_args()
config.num_clients = 15
config.clients_per_round = len(clients_this_round)
config.num_rounds_api_max = 12
config.dataset = 'mnist'
config.non_iid_type = 'dirichlet'
config.non_iid_alpha = 0.5
config.local_epochs = 1
config.client_optimizer = 'sgd'
config.client_lr = 0.01
config.seed = 42
config.quantity_skew_type = 'uniform'

# Etapa 2: Preencher estado global
FL_STATE['config'] = config
FL_STATE['history_log'] = collections.defaultdict(list)
tf.keras.utils.set_random_seed(config.seed)
np.random.seed(config.seed)

# Etapa 3: Carregar e distribuir dados
print("Carregando e particionando dados...")
load_and_distribute_data_api()

# Etapa 4: Criar modelo global
print("Criando modelo global...")

# Defina o número de execuções para cada configuração
num_executions = 5  # Número de execuções por configuração

# Loop para cada configuração de Pruning e Quantização
for config.prune, config.quanti in [('n', 'n'),('y', 'n'),('n', 'y'), ('y', 'y')]:
    print(f"\n--- Executando configuração: Pruning: {config.prune}, Quantização: {config.quanti} ---")
    
    # Inicializar listas para armazenar as médias das execuções por rodada
    round_losses = {round_num: [] for round_num in range(1, config.num_rounds_api_max + 1)}
    round_accuracies = {round_num: [] for round_num in range(1, config.num_rounds_api_max + 1)}
    round_test_losses = {round_num: [] for round_num in range(1, config.num_rounds_api_max + 1)}
    round_test_accuracies = {round_num: [] for round_num in range(1, config.num_rounds_api_max + 1)}

    for execution_num in range(1, num_executions + 1):
        model = create_model_api(config.dataset, num_classes_override=FL_STATE['num_classes'], seed=config.seed, config=config)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        global_weights = model.get_weights()
        FL_STATE['global_model'] = model

        # Etapa 4.5: Carregar os clientes salvos da simulação original
        selected_clients_per_round = load_selected_clients()
        print(f"\n### Execução {execution_num} ###")
        
        # Resetar o modelo global para garantir consistência entre as execuções
        model.set_weights(global_weights)
        
        compression_applied = False  # Reset compression flag for each configuration
        test_metrics_log = []  # Reset the test metrics log for each execution
        
        for round_num in range(1, config.num_rounds_api_max + 1):
            print(f"\n--- Rodada {round_num} ---")

            clients_this_round = selected_clients_per_round.get(round_num)
            if not clients_this_round or not isinstance(clients_this_round, list):
                print(f"⚠️ Clientes para a rodada {round_num} não encontrados ou formato inválido. Pulando.")
                continue

            print(f"→ {len(clients_this_round)} clientes selecionados para a rodada {round_num}.")
            config.clients_per_round = len(clients_this_round)

            client_updates = []
            sample_counts = []
            losses = []
            accs = []

            for client_id in clients_this_round:
                dataset = FL_STATE['client_train_datasets'][client_id]
                num_samples = FL_STATE['client_num_samples_unique'][client_id]

                updated_weights, loss, acc, _ = client_update_api(
                    model, global_weights, dataset, num_samples
                )

                client_updates.append(updated_weights)
                sample_counts.append(num_samples)
                losses.append(loss)
                accs.append(acc)

                print(f"Cliente {client_id} | Loss: {loss:.4f} | Acc: {acc:.4f} | Samples: {num_samples}")

            # --- Agregação do modelo global ---
            global_weights = aggregate_weights_fedavg_api(client_updates, sample_counts)
            model.set_weights(global_weights)

            print(f"📦 Tamanho do modelo (antes da compressão): {get_model_size_mb(model):.4f} MB")

            # Aplicar compressão
            if (config.prune == 'y' or config.quanti == 'y') and not compression_applied:
                print("→ Aplicando compressão no modelo global...")

                if config.prune == 'y' and config.quanti == 'n':
                    pruned_model = structurally_prune_model(
                        model,
                        dataset_name=config.dataset,
                        conv_prune_ratio=0.4,
                        dense_prune_ratio=0.5
                    )
                    model = pruned_model
                    model.compile(
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                    )

                # **Sempre aplica quantização quando config.quanti == 'y'**

                elif config.prune == 'y' and config.quanti == 'y':
                    pruned_model = structurally_prune_model(
                        model,
                        dataset_name=config.dataset,
                        conv_prune_ratio=0.4,
                        dense_prune_ratio=0.5
                    )
                    model = pruned_model
                    model.compile(
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                    )
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_model = converter.convert()
                    print(f"📦 Modelo podado e quantizado (TFLite): {len(tflite_model)/(1024**2):.4f} MB")
                    
                global_weights = model.get_weights()
                model.set_weights(global_weights)
                FL_STATE['global_model'] = model

                compression_applied = True
                
            if config.prune == 'n' and config.quanti == 'y':  # Aqui, sempre faz a quantização
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                print(f"📦 Modelo quantizado (TFLite): {len(tflite_model)/(1024**2):.4f} MB")
                    
                global_weights = model.get_weights()
                model.set_weights(global_weights)
                FL_STATE['global_model'] = model

            print(f"📦 Tamanho final do modelo global: {get_model_size_mb(model):.4f} MB")

            # --- Registro de resultados ---
            avg_loss = np.average(losses, weights=sample_counts)
            avg_acc = np.average(accs, weights=sample_counts)

            # Armazenar os resultados de cada execução por rodada
            round_losses[round_num].append(avg_loss)
            round_accuracies[round_num].append(avg_acc)

            print(f"> Média Loss: {avg_loss:.4f} | Média Acc: {avg_acc:.4f}")

            # Avaliação no dataset de teste
            if FL_STATE['centralized_test_dataset']:
                test_loss, test_acc = model.evaluate(FL_STATE['centralized_test_dataset'], verbose=0)
                FL_STATE['history_log']['global_test_loss'].append(test_loss)
                FL_STATE['history_log']['global_test_acc'].append(test_acc)
                print(f">> Avaliação Global | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
                test_metrics_log.append({
                    "execution": execution_num,
                    "round": round_num,
                    "test_loss": test_loss,
                    "test_acc": test_acc
                })
        
        # Calcular a média das execuções por rodada
        avg_round_losses = {round_num: np.mean(round_losses[round_num]) for round_num in round_losses}
        avg_round_accuracies = {round_num: np.mean(round_accuracies[round_num]) for round_num in round_accuracies}

        # Preparar os resultados finais para salvar
        avg_results = []
        for round_num in range(1, config.num_rounds_api_max + 1):
            avg_results.append({
                "round": round_num,
                "avg_loss": avg_round_losses[round_num],
                "avg_accuracy": avg_round_accuracies[round_num]
            })

        # Salvar os resultados em um único arquivo CSV
        df_avg_metrics = pd.DataFrame(avg_results)
        csv_filename = f"avg_results_prune_{config.prune}_quant_{config.quanti}.csv"
        df_avg_metrics.to_csv(csv_filename, index=False)
        print(f"📁 Médias das execuções salvas em: {csv_filename}")

print("\n✅ Fim do experimento federado.")
