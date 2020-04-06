import re
data = []
name = str(1)
dst = "outputs/semeval_output_pairs_" + name + ".txt"
tail_dict = {}
with open(dst, 'r') as file:
    for x in file.readlines():
        head, tail, score = x.strip().split('\t')
        head = re.sub(' ', '_', head)
        tail = re.sub(' ', '_', tail)
        data.append([head, tail])
        tail_dict[tail] = 1

output = "outputs/onto_" + name + ".ttl"
with open(output, 'w') as file:

    file.write("@prefix ex: <http://nandana.org/example/ontology/> ." + '\n')
    file.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> ." + '\n')
    file.write("@prefix owl: <http://www.w3.org/2002/07/owl#> ." + '\n')
    file.write("@prefix dc11: <http://purl.org/dc/elements/1.1/> ." + '\n')
    file.write("@prefix schema: <http://schema.org/> ." + '\n')

    for i in range(len(data)):
        file.write('ex:' + data[i][0] + ' a ' + 'rdfs:Class, owl:Class; rdfs:label "'+ data[i][0] + '" ; rdfs:subClassOf ' + 'ex:' + data[i][1] + ' .' + '\n')

print('Finished!')

# Tool
# http://rdfvalidator.mybluemix.net/