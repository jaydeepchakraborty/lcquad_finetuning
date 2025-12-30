
# inference on trained model (LCQUAD)
# test_text = {
#     "question": "Which languages does Odia speak?",
#     "org_sparql": "SELECT (COUNT(?sub) AS ?value ) { ?sub wdt:P1412 wd:Q33810 }"
# }
# lcquadmodel_helper.inference_lcquad_model(test_text)


# lcquad_model_helper_obj = LCQUADModelHelper(lcquad_conf)

# training gpt model (LCQUAD)
# lcquad_model_helper_obj.training_model()

# test on trained model (LCQUAD)
# lcquad_model_helper_obj.test_lcquad_model()

# inference on trained model (LCQUAD)
# test_text = {
#     "question": "Which languages does Odia speak?",
#     "org_sparql": "SELECT (COUNT(?sub) AS ?value ) { ?sub wdt:P1412 wd:Q33810 }"
# }
# test_text = {
#     "question": "At the time of 2.61e+06, what was the population of Brasilla?",
#     "org_sparql": "SELECT ?value WHERE { wd:Q2844 p:P1082 ?s . ?s ps:P1082 ?x filter(contains(?x,'2.61e+06')) . ?s pq:P585 ?value}"
# }
# test_text = {
#     "question": "What nearby city is the twin of Dusseldorg?",
#     "org_sparql": "SELECT ?answer WHERE { wd:Q1718 wdt:P47 ?answer . ?answer wdt:P190 wd:Q324941}"
# }
# test_text = {
#     "question": "Did Billy Graham die in Montreal?",
#     "org_sparql": "SELECT ?value WHERE { wd:Q213550 p:P20 ?s . ?s ps:P20 wd:Q736831 . ?s pq:P131 ?value}"
# }
# test_text = {
#     "question": "In which region does the Rideau Canal join the Ottawa River?",
#     "org_sparql": "SELECT ?value WHERE { wd:Q651323 p:P403 ?s . ?s ps:P403 wd:Q60974 . ?s pq:P131 ?value}"
# }
# test_text = {
#     "question": "When Jean Umansky was nominated for Amelie, what award was the nomination for?",
#     "org_sparql": "SELECT ?obj WHERE { wd:Q484048 p:P1411 ?s . ?s ps:P1411 ?obj . ?s pq:P2453 wd:Q6171615 }"
# }
# test_text = {
#     "question": "Name all the superpowers of Wonder Woman.",
#     "org_sparql": "SELECT (COUNT(?obj) AS ?value ) { wd:Q338430 wdt:P2563 ?obj }"
# }

# lcquad_model_helper_obj.inference_lcquad_model(test_text)
########################################################