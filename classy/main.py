import click


@click.group()
def cli():
    pass

@cli.command()
@click.argument('test', nargs=1)
def repr(test):
    print(test, ', sucesso demais repr')

@cli.command()
def anin():
    print("heheh")

def aninhado():
    print('sim, é possivel aninhar')

@cli.command()
@click.argument('test', nargs=1)
def e2e(test):
    print(test, ', sucesso demais etc')


if __name__ == "__main__":
    cli()


