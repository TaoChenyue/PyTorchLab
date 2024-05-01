# Pix2Pix
## [datasets](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
- [ ] cityscapes
- [ ] edges2handbags 
- [x] edges2shoes 
- [x] facades
- [ ] maps 
- [ ] night2day	 
## experiments
### edges2shoes
```shell
python main.py fit -c exps/pix2pix/configs/edges2shoes.yaml
```
input image:
![alt text](images/edges2shoes/input_image.png)
input generation:
![alt text](images/edges2shoes/input_generation.png)
output generation:
![alt text](images/edges2shoes/output_generation.png)
### facades
```shell
python main.py fit -c exps/pix2pix/configs/facades.yaml
```
input image:
![alt text](images/facades/input_image.png)
input generation:
![alt text](images/facades/input_generation.png)
output generation:
![alt text](images/facades/output_generation.png)
