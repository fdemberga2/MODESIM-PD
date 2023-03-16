from modsim import *
bikeshare = State(Makati = 9, Manila = 3)
def bike_to_makati():
    print('Moving to Makati')
    bikeshare.Makati +=1
    bikeshare.Manila -=1
def bike_to_manila():
    print('Moving to Manila')
    bikeshare.Makati -=1
    bikeshare.Manila +=1

if flip(0.7):
    print('Heads')
else:
    print('Tails')

bikeshare = State(Makati=9,Manila=3)
def step(p1,p2):
    if flip(p1):
        bike_to_makati()
    elif flip(p2):
        bike_to_manila
    else:
        print('No Movement')

results = TimeSeries()
results[0] = bikeshare.Makati
