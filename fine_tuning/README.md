# Fine-tuning (pipeline separado do app)

Este diretório descreve um pipeline simples de fine-tuning com LoRA.  
Ele roda **fora** do FastAPI e gera um modelo ajustado que depois pode ser
carregado no Ollama. O app continua o mesmo, só muda o modelo usado.

## Fluxo geral
1. Preparar dataset (JSONL com pares `prompt`/`response`).
2. Rodar treino LoRA (PEFT + Transformers).
3. Exportar o modelo ajustado.
4. Converter para GGUF (se necessário) e importar no Ollama.

## Estrutura
- `data/sample.jsonl`: exemplo mínimo de dataset.
- `prepare_dataset.py`: normaliza dados para o formato esperado.
- `train_lora.py`: script de treino LoRA (não executa aqui).
- `requirements.txt`: dependências do pipeline de treino.

## Exemplo de dataset (JSONL)
Cada linha é um JSON:
```
{"prompt": "Explique o IPCA de forma simples.", "response": "IPCA é o índice oficial de inflação..."}
```

## Como rodar (exemplo)
```
python fine_tuning/prepare_dataset.py \
  --input fine_tuning/data/sample.jsonl \
  --output fine_tuning/data/train.jsonl

python fine_tuning/train_lora.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --train_file fine_tuning/data/train.jsonl \
  --output_dir fine_tuning/output
```

## Depois do treino
1. Converta o modelo para GGUF (se for usar no Ollama).
2. Crie um `Modelfile` no Ollama apontando para o GGUF.
3. Ajuste `OLLAMA_MODEL` (ou `settings.ollama_model`) para o novo nome.
