from flask import Flask
from flask import Flask, render_template, flash, request , session , url_for , redirect
import joblib 
from forms import InputForm
import datetime;


app = Flask(__name__)
# Flask-WTF requires an encryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'


@app.route("/" , methods=['GET', 'POST'] )
def home():
    result = 2
    form = InputForm(request.form) 
    model = joblib.load('XGBoost_model')
    if request.method == 'POST':
        DISBURSED_AMOUNT = form.DISBURSED_AMOUNT.data
        ASSET_COST = form.ASSET_COST.data
        PRI_DISBURSED_AMOUNT = form.PRI_DISBURSED_AMOUNT.data
        PRI_ACTIVE_ACCTS = form.PRI_ACTIVE_ACCTS.data
        PRI_OVERDUE_ACCTS = form.PRI_OVERDUE_ACCTS.data
        PRIMARY_INSTAL_AMT = form.PRIMARY_INSTAL_AMT.data
        NEW_ACCTS_IN_LAST_SIX_MONTHS = form.NEW_ACCTS_IN_LAST_SIX_MONTHS.data
        DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS = form.DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS.data
        NO_OF_INQUIRIES = form.NO_OF_INQUIRIES.data
        LOANEE_DOB_DAYS_DAY = form.LOANEE_DOB_DAYS_DAY.data
        LOANEE_DOB_DAYS_Month = form.LOANEE_DOB_DAYS_Month.data
        LOANEE_DOB_DAYS_Year = form.LOANEE_DOB_DAYS_Year.data
        EMPLOYMENT_TYPE = form.EMPLOYMENT_TYPE.data
        Risk = form.Risk.data

        print(LOANEE_DOB_DAYS_DAY)
        print(int(LOANEE_DOB_DAYS_DAY))

        ct = datetime.datetime(int(LOANEE_DOB_DAYS_Year) , int(LOANEE_DOB_DAYS_Month), int(LOANEE_DOB_DAYS_DAY))
        ts = ct.timestamp()
        print(int(ts))
        print()

        in1 , in2 , in3 , in4  = 0 , 0 , 0 , 0
        if(Risk == 1 ):
            in1 , in2 , in3 , in4  = 1 , 0 , 0 , 0
        if(Risk == 2 ):
            in1 , in2 , in3 , in4  = 0 , 2 , 0 , 0
        if(Risk == 3 ):
            in1 , in2 , in3 , in4  = 0 , 0 , 1 , 0
        if(Risk == 4 ):
            in1 , in2 , in3 , in4  = 0 , 0 , 0 , 1


        All_data = [
            int(PRI_ACTIVE_ACCTS),
            int(PRI_OVERDUE_ACCTS),
            int(PRI_DISBURSED_AMOUNT), 
            int(PRIMARY_INSTAL_AMT),
            int(NEW_ACCTS_IN_LAST_SIX_MONTHS),
            int(DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS),
            int(NO_OF_INQUIRIES),
            int(int(str(ts)[:5])),
            int(int(DISBURSED_AMOUNT)/int(ASSET_COST)),
            int(EMPLOYMENT_TYPE),
            in1 , 
            in2 , 
            in3 , 
            in4 , 
        ]#--
        print(All_data)
        model = joblib.load('XGBoost_model')
        print(model.predict([All_data]))
        result = model.predict([All_data])[0]
        # result2 = model.predict([[inp1, inp2 , PRI_NO_OF_ACCTS]])
    return render_template("home.html"  , form=form , result = result )


@app.route("/page3" , methods=['GET', 'POST'] )
def page3():

    return "page3"

@app.route("/page2" , methods=['GET', 'POST'] )
def page2():
    # ct stores current time
    ct = datetime.datetime(2018, 9, 15)
    ts = ct.timestamp()


    return "page2"



    'PERFORM_CNS_DESC_Low', 'PERFORM_CNS_DESC_Medium',
    'PERFORM_CNS_DESC_High', 'PERFORM_CNS_DESC_Very high'