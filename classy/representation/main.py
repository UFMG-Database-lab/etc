import click

@click.group()
def cli():
    pass


#set a list of arguments
@click.option('--debug/--no-debug', default=False)

#set representation function
@cli.command('repr')  # @cli, not @click!
def representation(debug):
    click.echo('Test succeded!')


if __name__ == '__main__':
    cli()