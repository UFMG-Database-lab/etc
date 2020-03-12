import click
from os import path


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
            value = list(value)

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


"""Representation"""

#set a list of arguments
@click.option('-d', '--datasetdir', required=True, type=str, cls=OptionEatAll, help='Dataset path (For more info: readme.txt)')
@click.option('-ir', '--input-repr', type=str, cls=OptionEatAll, default=[], help='[Optional]')
@click.option('--silence', is_flag=True, help='Silence the bar.')
@click.option('-dsm','--dont-save-model', is_flag=True, default=True, help='To do not save the Vectorizer.')
@click.option('-s', '--seed', type=int, default=42, help='Seed to randomic generation.')
@click.option('-f', '--nfolds', type=str, default='10', help=f'Number of fold to build (if the folds are already made, the splits will be used).')
@click.option('-o', '--output', type=str, default=path.join('..','..','representations'), help=f'Path to the output directory (to save the splits and representations).')

#representation methods
@cli.command('repr')
def representation(datasetdir, input_repr, silence, dont_save_model, seed, nfolds, output):
    print(datasetdir, input_repr, silence, dont_save_model, seed, nfolds, output)



"""Classification"""

#set a list of arguments
@click.option('-r', '--representationdir', required=True, type=str, cls=OptionEatAll, help=' ')
@click.option('-ic', '--input-classifier', type=str, cls=OptionEatAll, default=['svm', 'lr'], help='Classifiers (can be an json file or some default implemented) [Default: [ svm, lr ] ]')
@click.option('--silence', is_flag=True, help='Silence the bar.')
@click.option('--verbose', is_flag=True, help='Show entire grid_results summs.')
@click.option('-cv', '--cross-val', type=int, default=5, help='Number of folds to cross val the train under grid_serach [IF APPLICABLE].')
# @click.option('-n-jobs', type=int, default=cpu_count(), help=f'Number of jobs to run the grid_serach [IF APPLICABLE, DEFAULT: ALL_CPUS ].')
@click.option('-dsm','--dont-save-model', is_flag=True, default=True, help='To do not save the Vectorizer.')
@click.option('-s', '--seed', type=int, default=42, help='Seed to randomic generation.')
@click.option('-o', '--output', type=str, default=path.join('..','..','..','classification'), help='Path to the output directory (to save the splits and representations).')

#classification methods
@cli.command('class')
def classification(representationdir, input_classifier, silence, verbose, cross_val, dont_save_model, seed, output):
    print(representationdir, input_classifier, silence, verbose, cross_val, dont_save_model, seed, output)



"""Evaluation"""

#set a list of arguments
@click.option('-r', '--result', required=True, type=str, cls=OptionEatAll)
# @click.option('-m','--metrics', type=str, cls=OptionEatAll, default=['f1_micro', 'f1_macro'], , type=click.Choice(METRICS.keys()), help='Metrics to measure the results.')
@click.option('-t','--tests', type=str, cls=OptionEatAll, default=None, help='Statistical test to eval the results.')
@click.option('-T','--times', is_flag=True, help='Show time analysis.')
@click.option('-s','--significance', type=float, nargs=1, default=0.05, help='Significance to statistical test.')
@click.option('--verbose', is_flag=True, help='Show extra analysis.')

#evaluation methods
@cli.command('eval')
def evaluation(result, tests, times, significance, verbose):
    print(result, tests, times, significance, verbose)



""" End-to-end"""

#set a list of arguments
@click.option('--dataset', '-d', required=True, cls=OptionEatAll, help='Dataset path (For more info: readme.txt)')
@click.option('-i', '--input-method', type=str, cls=OptionEatAll, default=[], help=f'[Optional] Input of the method descriptor.')
@click.option('--silence', is_flag=True, help=f'Silence the progress bar.')
@click.option('-sm','--save-model', is_flag=True, help=f'To save the Classifier object.')
@click.option('-pp','--predict-proba', default=False, help=f'Save the predicted probabilities to all class.')
@click.option('-s', '--seed', type=int, default=42, help=f'Seed to randomic generation.')
@click.option('-f', '--nfolds', type=str, default='10', help=f'Name of fold to build (if the folds are already made, the splits will be used).')
@click.option('-enc', '--encoding', type=str, default='utf8', help=f'Encoding to read/write dataset.')
@click.option('-o', '--output', type=str, default=path.join('..','..','..','e2e_result'), help=f'Path to the output directory (to save classification results).')
@click.option('-F', '--force', cls=OptionEatAll, default=None, help=f'Force fold to (re)execute.')

#e2e methods
@cli.command('e2e')
def e2e(dataset, input_method, silence, save_model, predict_proba, seed, nfolds, encoding, output, force):

    # print (dataset, input_method, silence, save_model, predict_proba, seed, nfolds, encoding, output, force)

    if force is not None:
        if 'all' in force:
            force = 'all'
        else:
            force = list(map(int, force))



if __name__ == '__main__':
    cli()