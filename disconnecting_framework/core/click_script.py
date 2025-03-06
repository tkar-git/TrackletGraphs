#Set up the click command line interface for the framework
#This script is executed when the user types "tracklet" in the command line
#The user can then type "tracklet discon" to run the infer command

import click
from . import test_stage, graph_construction_stage, disconnect_stage, eval_stage

@click.group()
def cli():
    pass


@cli.command()
def test():
    test_stage.main()

@cli.command()
@click.argument("config_file")
def build(config_file):
    graph_construction_stage.main(config_file)

@cli.command()
@click.argument("config_file")
def discon(config_file):
    disconnect_stage.main(config_file)


@cli.command()
@click.argument("config_file")
def eval(config_file):
    eval_stage.main(config_file)

if __name__ == '__main__':
    cli()