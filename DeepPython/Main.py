import sys
from DeepPython import Data, Bookmaker, Launch, Params, Elostd, Elostdid


print(sys.version)


data = Data.Data(Params.FILE)

bookm = Bookmaker.Bookmaker(data_dict=data.meta_datas, customizator={'normalized': True})
eloStd1 = Elostdid.Elostd(data_dict=data.meta_datas, customizator={'normalized': True})
eloStd2 = Elostd.Elostd(data_dict=data.meta_datas, customizator={'normalized': True})

dictModels = {"EloStd": eloStd1, "EloStd_id": eloStd2, "Bookmaker": bookm}

launch = Launch.Launch(data, dictModels, 'Shuffle', display=1)

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
