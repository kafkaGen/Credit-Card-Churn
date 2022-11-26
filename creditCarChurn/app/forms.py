from flask_wtf import FlaskForm
from wtforms import IntegerField, DecimalField, SelectField, SubmitField
from wtforms.validators import InputRequired, NumberRange


class DataForm(FlaskForm):
    Customer_Age = IntegerField(
        validators=[NumberRange(20, 120), InputRequired()])
    Gender = SelectField(choices=['M', 'F'], validators=[InputRequired()])
    Dependent_count = IntegerField(
        validators=[NumberRange(0, 10), InputRequired()])
    Education_Level = SelectField(choices=['College', 'Doctorate', 'Graduate', 'High School', 'Post-Graduate',
                                           'Uneducated', 'Unknown'], validators=[InputRequired()])
    Marital_Status = SelectField(
        choices=['Married', 'Single', 'Divorced', 'Unknown'], validators=[InputRequired()])
    Income_Category = SelectField(choices=['$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K',
                                           'Less than $40K', 'Unknown'], validators=[InputRequired()])
    Card_Category = SelectField(
        choices=['Blue', 'Gold', 'Platinum', 'Silver'], validators=[InputRequired()])
    Months_on_book = IntegerField(
        validators=[NumberRange(0, 60), InputRequired()])
    Total_Relationship_Count = IntegerField(
        validators=[NumberRange(0, 6), InputRequired()])
    Months_Inactive_12_mon = IntegerField(
        validators=[NumberRange(0, 6), InputRequired()])
    Contacts_Count_12_mon = IntegerField(
        validators=[NumberRange(0, 6), InputRequired()])
    Credit_Limit = DecimalField(validators=[NumberRange(0), InputRequired()])
    Total_Revolving_Bal = DecimalField(
        validators=[NumberRange(0), InputRequired()])
    Avg_Open_To_Buy = DecimalField(
        validators=[NumberRange(0), InputRequired()])
    Total_Amt_Chng_Q4_Q1 = DecimalField(
        validators=[NumberRange(0, 5), InputRequired()])
    Total_Trans_Amt = IntegerField(
        validators=[NumberRange(0), InputRequired()])
    Total_Trans_Ct = IntegerField(
        validators=[NumberRange(0, 150), InputRequired()])
    Total_Ct_Chng_Q4_Q1 = DecimalField(
        validators=[NumberRange(0, 4), InputRequired()])
    Avg_Utilization_Ratio = DecimalField(
        validators=[NumberRange(0, 1), InputRequired()])

    submit = SubmitField('Sign In')
