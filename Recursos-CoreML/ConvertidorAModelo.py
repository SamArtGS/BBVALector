import coremltools
from sklearn.linear_model import LinearRegression
import pandas

primaryData = pandas.read_csv("HouseSalesInCA.csv")

modelo = LinearRegression()
modelo.fit(primaryData[["Bedrooms","Bathrooms","Size"]], primaryData["Price"])

coremlModel = coremltools.converters.sklearn.convert(modelo, ["Bedrooms","Bathrooms","Size"], "Price")

coremlModel.author = "Gabo Cuadros - Un Swifter"
coremlModel.short_description = "Este modelo calculara el precio de una casa dependiendo el numeor de habitaciones, banos y terreno en el estado de California"
coremlModel.license = "MIT"

coremlModel.save("HouseSalesInCA.mlmodel")
