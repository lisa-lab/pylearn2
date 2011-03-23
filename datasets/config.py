global resolvers

def resolve(d):
    tag = framework.config.get_tag(d)

    if tag != 'dataset':
        raise TypeError('framework.datasets.config asked to resolve a config dictionary with tag "'+tag+'"')

    t = framework.config.get_str(d,'typename')

    try:
        resolver = resolvers[t]
    except:
        raise TypeError('framework.datasets does not know of a dataset type "'+t+'"')

    return resolver(d)


def resolve_avicenna(d):
    import framework.datasets.avicenna
    return framework.config.checked_call(framework.datasets.avicenna.Avicenna,d)

resolvers = {
            'avicenna' : resolve_avicenna
        }

import framework.config
