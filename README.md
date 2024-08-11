# Various Reinforcement Learning Algorithms on Racetrack Simulations
We are presented with 3 different racetracks, an L track, an O track and an R track for our racecar to travel in. In this project, we build and test several factors to drive an optimal race. Firstly we build an environment for the car to travel in, this involves detecting if the car has crossed the finish line or not, or has crashed into a wall. To determine whether a car has crashed into a wall, we use Bresenham’s line algorithm to track our movement from point a to b and check if any of those points contains a wall. To determine the closest on-track point to a crash site, we start from the crash site and use breadth-first search via a queue to determine the closest on-track coordinate to the crash site.

## Run it
```
python main.py
```

## Read it! 
```
writeup.pdf
```
