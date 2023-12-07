# Treinando YoloV8
Fazendo um treinamento da YoloV8

## Treinando
### Instalando pytorch com CUDA para uso da GPU
Se você tem uma placa de vídeo compatível com CUDA, primeiro instale o PyTorch com CUDA neste link

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Instale o Yolo
```
pip install ultralitics
```

### Treinamento
Para treinar, acompanhe o vídeo e use o arquivo "train_among_v8.py"

## Testando
Para testar, se quiser fazer o rastreio coloque a variavel "seguir" em True
Para desenhar o rastreio coloque a variavel "deixar_rastro" em True

### Testando com WebCam
Para testar com WebCam use o arquivo "detectar_usando_webcam.py"

### Testando Capturando Tela
Para testar capturando a tela use o arquivo "detectar_capturando_tela.py"

Neste caso configure o tamanho da tela no campo "size" da wincap e ajuste o offset do ponto inicial que quer capturar

Também é possível capturar passando o nome da janela que deseja usar (porém nem sempre funciona)


Valeuuu

[!["Treinando Redes Neurais Com Imagens Próprias"](https://img.youtube.com/vi/KV5lszcKuiE/0.jpg)](https://www.youtube.com/watch?v=KV5lszcKuiE)
