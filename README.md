# HTTP YOLO Server

Este projeto consiste em um servidor HTTP implementado utilizando apenas a biblioteca de sockets nativa do Python. O objetivo principal é demonstrar a integracao de conceitos de redes de computadores com visao computacional em tempo real.

## Caracteristicas do Projeto

- Servidor HTTP Nativo: Implementacao completa de parsing de headers, roteamento e manuseio de corpo de requisicoes sem o uso de frameworks web (Flask, FastAPI, etc).
- Streaming MJPEG: Transmissao de video em tempo real utilizando o protocolo multipart/x-mixed-replace.
- Motor de Visao YOLO: Integracao com modelos YOLO (.pt e .onnx) para deteccao de objetos em tempo real no computador host.
- Gestao Dinamica de Modelos: O servidor escaneia automaticamente a pasta de modelos no disco e os disponibiliza na interface sem necessidade de configuracao manual.
- Captura de Midia Local: Funcionalidade de download de fotos e gravacoes de video diretamente para a pasta de Downloads do dispositivo do cliente (celular ou PC).

## Requisitos de Sistema

- Python 3.8 ou superior
- OpenCV (cv2)
- NumPy
- Ultralytics (para suporte a modelos .pt)

## Estrutura de Diretorios

- http_server.py: Nucleo do servidor socket e roteamento HTTP.
- vision_engine.py: Gerenciamento da camera, processamento de frames e inferencia YOLO.
- server_config.py: Configuracoes globais e estado da aplicacao.
- index.html: Interface web premium com suporte a dispositivos moveis.
- models/: Diretorio onde os arquivos de modelos (.pt ou .onnx) devem ser colocados.

## Como Executar

1. Instale as dependencias necessarias:
   pip install opencv-python numpy ultralytics

2. Coloque seus modelos YOLO na pasta models/.

3. Inicie o servidor:
   python http_server.py

4. Acesse o sistema atraves do navegador:
   Local: http://localhost:8080
   Rede Local: http://[IP-DO-HOST]:8080

## Observacoes de Seguranca e Redes

Este servidor utiliza HTTP puro para fins didaticos. Em navegadores modernos, o acesso a recursos de hardware (como a camera do cliente) é restrito a contextos seguros (HTTPS ou localhost). Para demonstracoes em rede local utilizando a camera de dispositivos moveis, recomenda-se o uso de um tunel TLS (como o ngrok) ou a configuracao de excecoes de seguranca no navegador (chrome://flags).

## Autor
Projeto desenvolvido para a disciplina de Redes de Computadores.