import click
import sys


class OptionEatAll(click.Option):

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):

        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


@click.group()
def cli():
    pass

""" Evaluation code"""
#set a list of arguments

@click.option('--dataset', '-d', required=True, cls=OptionEatAll, help='Dataset path (For more info: readme.txt)')
@click.option('-i', '--input-method', type=str, cls=OptionEatAll, default=[], help=f'[Optional] Input of the method descriptor.')
# @click.option('--silence', required=False, default=False, nargs=1, help=f'Silence the progress bar.')
# @click.option('-sm','--save-model', default=False, nargs=0, help=f'To save the Classifier object.')
@click.option('-pp','--predict-proba', default=False, help=f'Save the predicted probabilities to all class.')
@click.option('-s', '--seed', type=int, default=42, help=f'Seed to randomic generation.')
@click.option('-f', '--nfolds', type=str, default='10', help=f'Name of fold to build (if the folds are already made, the splits will be used).')
@click.option('-enc', '--encoding', type=str, default='utf8', help=f'Encoding to read/write dataset.')
# @click.option('-o', '--output', type=str, default=click.Path('..','..','..','e2e_result'), help=f'Path to the output directory (to save classification results).')
@click.option('-F', '--force', cls=OptionEatAll, default=None, help=f'Force fold to (re)execute.')


#set representation function
@cli.command('e2e')
def evaluation(dataset, input_method, silence, save_model, predict_proba, seed, nfolds, encoding, force):
    click.echo(click.style('Sucess', fg='magenta'))
    print(silence, save_model)
    print (sys.argv[1:])



if __name__ == '__main__':
    cli()