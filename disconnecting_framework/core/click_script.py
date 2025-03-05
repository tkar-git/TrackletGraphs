#Set up the click command line interface for the framework
#This script is executed when the user types "tracklet" in the command line
#The user can then type "tracklet discon" to run the infer command

import click
from . import testing, disconnect

@click.group()
def cli():
    pass


@cli.command()
def test():
    testing.main()

@cli.command()
@click.argument("config_file")
def discon(config_file):
    disconnect.main(config_file)


if __name__ == '__main__':
    cli()