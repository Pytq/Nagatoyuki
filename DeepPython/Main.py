import sys
from DeepPython import Data, Bookmaker, Launch, Params, Elostd, Elostdid, Regresseur


print(sys.version)


data = Data.Data(Params.FILE)

bookm = Bookmaker.Bookmaker(data_dict=data.meta_datas, customizator={'normalized': True}, name='bookm')
regr = Regresseur.Regresseur(data_dict=data.meta_datas, name='regr')
elo = Elostdid.Elostd(data_dict=data.meta_datas, name='eloStd')


dictModels = {"Regresseur": regr, "EloStd": elo, "Bookmaker": bookm}

launch = Launch.Launch(data, dictModels, 'Lpack', display=1)

launch.execute()



"""

1)check_flow_slices

Data=data.data(“ForSlices”) 

Data_Descripteurs = Data.Getdescripteurs(“x.txt”, etc…) 

Slices=Slices.slices(Data_Descripteurs)

Et Data.ConstructXyz(« details ») 

Model= Model.model(Data_Descripteurs)

Cost = Cost.cost()

Optimisation = Optimisation.optimisation()

Launch(Data, Slices, Model, Cost, Optimisation)



Ce qu il faut encore faire :
   
 -> Clean le Launch (vieux copié collé degueu)
 
 -> Reflechir a que Data soit nesté dans Slices ?
 
 -> Choisir le mode de lecture et implémenter l'autre (genre random pour opti params)
 
 -> Reflechir a l archi si ce qu on veut faire est plus complexe que ca 
    (Genre composés de modèles, join datas etc)
 
 -> Separer toute la logique TF 
     (Comment ? Passer par delegues et refacto)
     (Pourquoi ? Parce que code spécifique, si on veut changer, pas easy)
     
 

"""
