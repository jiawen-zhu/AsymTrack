def count_param(model, include=None):
    count = 0
    for name, weight in model.items():
        if include in name or include=='all':
            # print(name)
            count+=weight.reshape(-1).size()[0]
    return count
