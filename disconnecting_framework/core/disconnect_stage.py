import yaml

def main(config_file):
    print("Entered disconnect stage")
    print('Modification')
    print('Yet another modification')
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
