import sys
from DeepPython import Data, Bookmaker, Launch, Params, Elostd, Elostdid, Regresseur, Elostdid_regr
import tensorflow as tf
import Stats

print(sys.version)

file = '../Output/output_new_2017_01_22_18_40_48'

# stats = Stats.Stats(file, s=10.)

session = tf.Session()
data = Data.Data(Params.FILE)

#bookm = Bookmaker.Bookmaker(data_dict=data.meta_datas, customizator={'toNormalize': True}, name='bookm', session=session)
#regr = Elostdid_regr.Elostdid_regr(data_dict=data.meta_datas, customizator={'trainit': True}, name='regr', session=session)
elo = Elostdid.Elostid(data_dict=data.meta_datas, customizator={'trainit': True}, name='eloStd', session=session)

#dictModels = {"Bookmaker": bookm, "Elo" : elo, "Regresseur": regr}
dictModels = {"Elo" : elo}
init_dict = Params.paramStd

launch = Launch.Launch(data, dictModels, 'Shuffle', display=1, init_dict=init_dict)

launch.grid_search('Elo')



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
