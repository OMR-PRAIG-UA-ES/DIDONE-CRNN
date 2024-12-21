# Generate didone dataset
1. Access to the folder `datasets/didone/`
2. Execute the file `obtain_data.py`
3. Return to project home folder

# To build image
```docker
docker build -t smp_bld .
```

# To run image
```docker
docker run -itd -v <path_to_repo>/simple/:/workspace --shm-size 24Gb --gpus device=0 --name simple smp_bld
```

Take into consideration that `<path_to_repo>`stands for the ubication where you have decided to clone this repository.

# To train

To specify type of training (standard or split encoding) modify directly the train.py file.

```shell
python train.py
```

Models will be saved in a folder called `models`.

# Changes to speed-up training
1. Change input height
  By default, the input image height is 128. You can reduce this by changing line 89 of the `train.py`file...
```python
model = CRNN(num_channels=1, img_height=128, output_size=len(dataset.vocab.c2i)).to(
```
... and line 45 of the `dataset.py` file...
```python
ResizeByHeight(128),
```
2. ...

Further changes will be appended to this list
