from os.path import exists



def write_results(results, path):
    
    file_exists = exists(path)
    if not(file_exists):
        with open(path,"a+") as f:
            f.write(results)
            f.write("\n")
    else:
        with open(path,"a+") as f:
            f.write(results)
