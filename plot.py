import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

from collections import defaultdict
from pygal_maps_world.maps import World

import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report




def yearPub():
    mast = pd.read_csv("tab_Mast.csv")
    year_Catag = defaultdict(int)

    xAxis = []
    yAxis = []

    for year in mast["Year"]:
        year_Catag[year] +=1

    for year in year_Catag:
        xAxis.append(year)

    for amt in year_Catag.values():
        yAxis.append(amt)

    plt.title('Yearly Publication')
    plt.plot(xAxis, yAxis)
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.show()


def yearCit():
    mast = pd.read_csv("tab_Mast.csv")
    mast = mast.drop_duplicates(subset= "Article No.", keep= 'first', inplace=False)

    cit_Catag = defaultdict(int)

    xAxis = []
    yAxis = []

    rowIndex = 0
    for year in mast["Year"]:
        cit_Catag[year] += mast.iloc[rowIndex, 6]
        rowIndex += 1

    for yearX in cit_Catag:
        xAxis.append(yearX)

    for amtY in cit_Catag.values():
        yAxis.append(amtY)

    plt.title('Yearly Citations')
    plt.plot(xAxis, yAxis)
    plt.xlabel('Year')
    plt.ylabel('Number of Citations')
    plt.show()

def countryPlot():
    mast = pd.read_csv("tab_Mast.csv")
    mast.drop(mast[(mast['Country'] == "0")].index, inplace=True)
    country_Catag = defaultdict(int)
  
    for countries in mast["Country"]:
        country_Catag[countries] += 1

    worldmap =  World()
    worldmap.title = 'Countries' 
    worldmap.add('No. of Publications', {
        counConvert("USA") : country_Catag.get("USA"), 
        counConvert("Cyprus") : country_Catag.get("Cyprus"), 
        counConvert("United Kingdom") : country_Catag.get("United Kingdom"), 
        counConvert("United Arab Emirates") : country_Catag.get("United Arab Emirates"), 
        counConvert("Taiwan") : country_Catag.get("Taiwan"), 
        counConvert("Denmark") : country_Catag.get("Denmark"), 
        counConvert("Canada") : country_Catag.get("Canada"), 
        counConvert("Spain") : country_Catag.get("Spain"),
        counConvert("China") : country_Catag.get("China"),
        counConvert("New Zealand") : country_Catag.get("New Zealand"),
        counConvert("Chile") : country_Catag.get("Chile"),
        counConvert("Italy") : country_Catag.get("Italy"),
        counConvert("Australia") : country_Catag.get("Australia"),
        counConvert("Israel") : country_Catag.get("Israel"),
        counConvert("France") : country_Catag.get("France"),
        counConvert("Germany") : country_Catag.get("Germany"),
        counConvert("India") : country_Catag.get("India"),
        counConvert("Slovakia") : country_Catag.get("Slovakia"),
        counConvert("Ireland") : country_Catag.get("Ireland"),
        counConvert("Kyrgyzstan") : country_Catag.get("Kyrgyzstan"),
        counConvert("Malaysia") : country_Catag.get("Malaysia"),
        counConvert("Pakistan") : country_Catag.get("Pakistan"),
        counConvert("Liechtenstein") : country_Catag.get("Liechtenstein"),
        counConvert("Norway") : country_Catag.get("Norway"),
        counConvert("Hong Kong") : country_Catag.get("Hong Kong"),
        counConvert("Korea") : country_Catag.get("Korea"),
        counConvert("Switzerland") : country_Catag.get("Switzerland"),
        counConvert("Mexico") : country_Catag.get("Mexico"),
        counConvert("Ukraine") : country_Catag.get("Ukraine"),
        counConvert("South Africa") : country_Catag.get("South Africa"),
        counConvert("Greece") : country_Catag.get("Greece"),
        counConvert("Russia") : country_Catag.get("Russia"),
        counConvert("Czech Republic") : country_Catag.get("Czech Republic"),
        counConvert("Palestine") : country_Catag.get("Palestine"),        
    })
    worldmap.render_to_file('abc.svg')

def counConvert(str):
    if (str == "USA"):
        return "us" 
    if (str == "Cyprus"):
        return "cy"
    if (str == "United Kingdom"):
        return "gb"
    if (str == "United Arab Emirates"):
        return "ae"
    if (str == "Taiwan"):
        return "tw"
    if (str == "Denmark"):
        return "dk"
    if (str == "Canada"):
        return "ca"
    if (str == "Spain"):
        return "es"
    if(str == "China"):
        return "cn"
    if (str == "New Zealand"):
        return "nz"
    if (str == "Chile"):
        return "cl"
    if (str == "Italy"):
        return "it"
    if (str == "Australia"):
        return "au"
    if (str == "Israel"):
        return "il"
    if (str == "France"):
        return "fr"
    if (str == "Germany"):
        return "de"
    if (str == "India"):
        return "in"
    if (str == "Slovakia"):
        return "sk"
    if (str == "Ireland"):
        return "ie"
    if (str == "Kyrgyzstan"):
        return "kg"
    if (str == "Malaysia"):
        return "my"
    if (str == "Pakistan"):
        return "pk"
    if (str == "Liechtenstein"):
        return "li"
    if (str == "Norway"):
        return "no"
    if (str == "Hong Kong"):
        return "hk"
    if (str =="Korea"):
        return "kr"
    if (str == "Switzerland"):
        return "ch"
    if (str == "Mexico"):
        return "mx"
    if (str == "Ukraine"):
        return "ua"
    if (str == "South Africa"):
        return "za"
    if (str == "Greece"):
        return "gr"
    if (str == "Russia"):
        return "ru"
    if (str == "Czech Republic"):
        return "cz"
    if (str == "Palestine"):
        return "ps"

def regression():
    mast = pd.read_csv("data.csv").fillna(0)
    le = LabelEncoder()

    le.fit(mast['Purchase']) #le.fit transform all the values of the column to either true (1) or false(0)
    mast['Purchase'] = le.transform(mast['Purchase']) #Did it to two columns where it could be applied
    le.fit(mast['Gender'])
    mast['Gender'] = le.transform(mast['Gender'])

    results = mast.corr(method='pearson')['SUS'].sort_values() #runs a the Pearson correlation test on all the column values in relation to the SUS column values
    print(results) #goes from -1 to 1, with -1 meaning a strong negative relationships (as one value goes up, the other goes down) 
    #1 means a strong positive relationship (as one value goes up, so does the other)
    #The signs indicate the state of correlation, positive meaning they are direct, and negative meaning they are inverse
    #0 means there is no correlation between values, anything in between -1 to 0 and 0 to 1 indicates the strength of their correlation

    y = mast['SUS'] #stores the SUS column data values as the dependent variable
    x = mast.drop(columns='SUS') #exclues the SUS column and treats every other one like an independent variable and its role in affecting the dependent variable (SUS)

    x = sm.add_constant(x) #add constants to the independent variable to clear up and take care of any sort of bias in the data. Helps clean up and set the data to 
    #be regressed

    model = sm.OLS(y, x).fit() #it takes the x and y variable which have been set up for with indepedent and dependent variable and puts it into an OLS model
    #In short, what it's accomplishing is that it's estimating coefficients in a linear regression model
    #This is to help find and describe the relationship between the predictor variable (independent variable) and response variable (SUS)
    #It is used to predict the value of the response(dependent) variable based off of the predictor (independent) variable
    print (model.summary())

    x = mast.drop(columns='SUS')
    y = mast['SUS']

    x_train, x_test, y_train, y_test = train_test_split(x, y) #splits data into train and test sets 
    #train set data is used for our model to make predictions and approximations
    #test set data is used to compare our model predictions to it and see if the model is working correctly
    #train set should always be bigger than test sets

    #//The R square regression analysis
    lr = LinearRegression().fit(x_train,y_train)  #Plots a line that determines the relationship between the independent and dependent variable 
    #Is used to predict the value of a variable (dependent) based off the value of another variable (independent)
    y_train_pred = lr.predict(x_train) #Used the linear regression line to predict values based off the train and test set data value
    y_test_tred = lr.predict(x_test) 

    print("The R square score of linear regression model is: ", lr.score(x_test,y_test)) #It shows how well our data from the regression model can explain observed or test data in this case
    #It shows how well the variability observed is explained or fitted by the regression model
    #//The R square regression analysis: end

    #//The 2-order Polynomial regression analysis
    quad = PolynomialFeatures (degree = 2)  #It is similar to what linear regression does but allows for a curved line instead. Works out for some relationship where
    #it is not linear at all. Allows for more flexibility and captures more variance in data points and relationships
    x_quad = quad.fit_transform(x) #transforms all the independent variable to this

    X_train,X_test,Y_train,Y_test = train_test_split(x_quad,y, random_state = 0)

    plr = LinearRegression().fit(X_train,Y_train) #plots the curved line

    Y_train_pred = plr.predict(X_train)
    Y_test_pred = plr.predict(X_test)

    print ("The R square score of 2-order polynomial regression model is: ", plr.score(X_test,Y_test)) 
    #//The 2-order Polynomial regression analysis: end

def classification():
    mast = pd.read_csv("data.csv").fillna(0)

    mast['Purchase'] = mast['Purchase'].fillna(mast['Purchase'].mean()) #cleans up data by filling in missing values with the mean value of that column
    mast['SUS'] = mast['SUS'].fillna(mast['SUS'].mean())
    mast['Duration'] = mast['Duration'].fillna(mast['Duration'].mean())
    mast['ASR_Error'] = mast['ASR_Error'].fillna(mast['ASR_Error'].mean())
    mast['Gender'] = mast['Gender'].fillna(mast['Gender'].mean())
    mast['Intent_Error'] = mast['Intent_Error'].fillna(mast['Intent_Error'].mean())

    y = mast['Purchase'].to_numpy() #setting up and cleaning data to be inserted into the model
    x = mast.drop('Purchase', axis = 1).to_numpy()

    #np.seterr(divide='ignore', invalid='ignore')

    scale = StandardScaler() #scales data to ensure normal distribution
    scaled_X = scale.fit_transform(x) #stores scaled data

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.4) #splits the data set into 70/30
    # 70% is to train the model while the rest is used to see how well the model performed

    oversample = SMOTE() #SMOTE is an oversampling method to solve imbalance problems. 
    over_sampled_X_train, over_sampled_y_train = oversample.fit_resample(X_train, y_train) 
    #It balances class distribution by randomly increasing minority class examples by replicating them

    lc = LogisticRegression() #sets up the classification to be tested
    svc = SVC(probability=True)
    nbc = GaussianNB()
    rfc = RandomForestClassifier()

    lc.fit(over_sampled_X_train, over_sampled_y_train) #fits the data into those models
    svc.fit(over_sampled_X_train, over_sampled_y_train)
    nbc.fit(over_sampled_X_train, over_sampled_y_train)
    rfc.fit(over_sampled_X_train, over_sampled_y_train)

    y_lc_predicted = lc.predict(X_test) #the results of the data being ran through the model being stored in an array
    y_lc_pred_proba = lc.predict_proba(X_test) #returns the an array of lists containing class probabilities for the input data points

    y_svc_predicted = svc.predict(X_test)
    y_svc_pred_proba = svc.predict_proba(X_test)

    y_nbc_predicted = nbc.predict(X_test)
    y_nbc_pred_proba = nbc.predict_proba(X_test)

    y_rfc_predicted = rfc.predict(X_test)
    y_rfc_pred_proba = rfc.predict_proba(X_test)    

    print('Linear Regression: \n', classification_report(y_test, y_lc_predicted)) #prints out in a table how well the model performed compared to the test results
    print('SVC: \n', classification_report(y_test, y_svc_predicted))
    print('Naive Bayes: \n', classification_report(y_test, y_nbc_predicted))
    print('Random Forest: \n', classification_report(y_test, y_rfc_predicted))

    models = ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes Classifier', 'Random Forest Classifier']
    predictions = [y_lc_predicted, y_svc_predicted, y_nbc_predicted, y_rfc_predicted] #Makes two different array to store the values which is to be put in the confusion matrix
    pred_probabilities = [y_lc_pred_proba, y_svc_pred_proba, y_nbc_pred_proba, y_rfc_pred_proba]

    for model, prediction, pred_proba in zip(models, predictions, pred_probabilities): #For loops to plot out the matrix for each classification model
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test.ravel(), prediction))
        disp.plot(
            include_values=True,
            cmap='gray',
            colorbar=False
        )
        disp.ax_.set_title(f"{model} Confusion Matrix")

    plt.figure(figsize=(30, 15)) #More labels to help understand the matrix
    plt.suptitle("ROC Curves")
    plot_index = 1

    for model, prediction, pred_proba in zip(models, predictions, pred_probabilities): #plots out the ROC curve to test model data
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:, 1])
        auc_score = auc(fpr, tpr)
        plt.subplot(3, 2, plot_index)
        plt.plot(fpr, tpr, 'r', label='ROC curve')
    # pyplot.figure(figsize=(5, 5))
        plt.title(f'Roc Curve - {model} - [AUC - {auc_score}]', fontsize=14)
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        plt.legend()
        plot_index += 1
    plt.show()