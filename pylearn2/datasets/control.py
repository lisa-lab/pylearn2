
load_data = [ True ]

def pop_load_data():
    global load_data

    del load_data[-1]

def push_load_data(setting):
    global load_data

    load_data.append(setting)

def get_load_data():
    return load_data[-1]
