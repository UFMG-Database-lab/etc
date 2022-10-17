import importlib

def from_descriptor(descriptor):
    if 'type' not in descriptor or descriptor['type'] == 'skl':
        init_params = {}
        if 'init_params' in descriptor:
            init_params = descriptor['init_params']
        assert 'classpath' in descriptor, 'Classpath should be in the descriptor.'
        return build_obj(descriptor['classpath'], init_params)
    if descriptor['type'] == 'skl-pipe':
        from sklearn.pipeline import make_pipeline
        pipeline = map(from_descriptor, descriptor['pipeline'])
        return  make_pipeline(*pipeline)
    if descriptor['type'].startswith('hug'):
        pass
    raise Exception(f"Descriptor type not found ({descriptor['type']}). Options [skl, skl-pipe, huggingface]")


def build_obj(classpath, init_params):
    module_name, class_name = classpath.rsplit('.', 1)
    module_obj = importlib.import_module(module_name)
    class_obj = module_obj.__getattribute__(class_name)
    return class_obj(**init_params)