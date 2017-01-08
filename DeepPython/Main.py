import Data
import Elostd
import Launch
import Params
import sys
import Bookmaker

print(sys.version)


data = Data.Data(Params.FILE + '.txt', typeslices='overtime')

model = Bookmaker.Bookmaker(data_dict=data.dict_dataToModel)

launch = Launch.Launch(data, model)

launch.Go()

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
