from math import sqrt
import random

MAP_SIZE = 200


def euclidean(x:tuple, y:tuple):
    return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def generate_cities(count:int):
    if count <20 or count > 40:
        return []
    cities = []
    for i in range(count):
        x = random.randrange(0, MAP_SIZE)
        y = random.randrange(0, MAP_SIZE)
        while (x,y) in cities and cities != []:
            x = random.randrange(0, MAP_SIZE)
            y = random.randrange(0, MAP_SIZE)
        cities.append((x,y))
    return cities




def main():
    cities = generate_cities(20)
    print(cities)
    return

if __name__ == '__main__':
    main()

