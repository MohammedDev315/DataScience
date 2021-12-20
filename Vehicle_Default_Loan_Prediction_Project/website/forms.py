from wtforms import Form, RadioField, StringField, PasswordField, validators , SubmitField

class InputForm(Form):
     DISBURSED_AMOUNT = StringField('DISBURSED_AMOUNT')
     ASSET_COST = StringField('ASSET_COST')
     PRI_DISBURSED_AMOUNT = StringField('PRI_DISBURSED_AMOUNT')
     PRI_ACTIVE_ACCTS = StringField('PRI_ACTIVE_ACCTS')
     PRI_OVERDUE_ACCTS = StringField('PRI_OVERDUE_ACCTS')
     PRI_CURRENT_BALANCE = StringField('PRI_CURRENT_BALANCE')
     PRIMARY_INSTAL_AMT = StringField('PRIMARY_INSTAL_AMT')
     NEW_ACCTS_IN_LAST_SIX_MONTHS = StringField('NEW_ACCTS_IN_LAST_SIX_MONTHS')
     DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS = StringField('DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS')
     NO_OF_INQUIRIES = StringField('NO_OF_INQUIRIES')
     LOANEE_DOB_DAYS_DAY = StringField('LOANEE_DOB_DAYS DAY')
     LOANEE_DOB_DAYS_Month = StringField('Month')
     LOANEE_DOB_DAYS_Year = StringField('Year')
     DISBURSAL_DATE_DAYS_Day = StringField('DISBURSAL_DATE_DAYS_Day')
     DISBURSAL_DATE_DAYS_Month = StringField('Month')
     DISBURSAL_DATE_DAYS_Year = StringField('Year')
     EMPLOYMENT_TYPE = RadioField(choices = [(  0 ,  'EMPLOYED' ), (1,  'SELF-EMPLOY')])
     Risk = RadioField(choices = [(  1 ,  'Low' ), ( 2 ,  'Medium'), ( 3 , 'High'), ( 4 , 'Very High')])
     submitButton = SubmitField('Submit')




