# Investigating the Viability of Generating Trading Strategies with Multi-Objective Evolutionary Algorithms
# ECM3401 - Source Code

## Building the docker images

Navigate into the EasyRide-1 directory then enter the following commands.

```shell
$ cd driver-microservice
$ docker build --tag driver_env .
$ cd ..
$ cd cost-microservice
$ docker build --tag cost_env .
$ cd ..
$ cd mapping-microservice
$ docker build --tag mapping_env .
$ cd ..

```

## Creating the docker network

```shell
$ docker network create --subnet 192.168.1.0/24 easy_ride_network 
```

## Running the the isolated containers

```shell
$ docker run --name container1 --net easy_ride_network \
        --ip 192.168.1.8 --detach \
        --publish 3000:8888 \
        --security-opt apparmor=unconfined driver_env

$ docker run --name container2 --net easy_ride_network \
        --ip 192.168.1.7 --detach \
        --publish 3001:8888 \
        --security-opt apparmor=unconfined cost_env

$ docker run --name container3 --net easy_ride_network \
        --ip 192.168.1.6 --detach \
        --publish 3002:8888 \
        --security-opt apparmor=unconfined mapping_env
```

## Dont't forget to remove the images, containers and network after use

```shell
$ docker kill container1
$ docker kill container2
$ docker kill container3
$ docker rm container1
$ docker rm container2
$ docker rm container3
$ docker network rm easy_ride_network
$ docker rmi driver_env
$ docker rmi cost_env
$ docker rmi mapping_env
```

# Generating a JWT token for testing a custom driver

I have included another file, generateToken.go, which can generate JWT tokens to allow for testing if you would like to test this feature. Run the generateToken.go file by entering:

```shell
$ go run generateToken.go
```

This will print to the terminal the JWT token that a driver can then send in the header
of a POST which allows the driver to be authenticated for creating, updating and deleting.
