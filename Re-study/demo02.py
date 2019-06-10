import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Square",type=int,help = "echo the string you use here")
args = parser.parse_args()
print(args.Square**2)
