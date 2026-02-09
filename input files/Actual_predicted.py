#!/usr/bin/env python
# coding: utf-8

# In[274]:


get_ipython().run_line_magic('matplotlib', 'inline')
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[67]:


# Load dataset
df = pd.read_csv("C:/Users/Grancy/Downloads/Salary_Data (1).csv")
df


# In[69]:


df.info()


# In[71]:


df.describe()


# In[73]:


# Checking for null data

df.isnull().sum()


# In[75]:


# Dropping null values from database

df.dropna(inplace=True)


# In[77]:


# Compute unique value counts for each feature variable

unique_counts = df.nunique()
unique_counts


# In[79]:


cols = ['Gender', 'Age', 'Education Level', 'Job Title', 'Years of Experience']


# In[109]:


for col in ['Gender', 'Education Level']:
    counts = df[col].value_counts()
    explode = [0.1] * len(counts)  # Explode all slices
    colors = sns.color_palette("pastel", len(counts))  # Use pastel colors

    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
        explode=explode, colors=colors, shadow=True, textprops={'fontsize': 12}
    )

    plt.title(f'{col} Distribution', fontsize=14, fontweight='bold')

    # Position the legend outside the pie chart
    plt.legend(wedges, counts.index, title=col, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

    plt.show()


# In[139]:


plt.figure(figsize=(10, 6))

# Count occurrences of each job title
job_counts = df['Job Title'].value_counts()

# Plot pie chart
plt.pie(job_counts, labels=job_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette("husl", len(job_counts)),
        wedgeprops={'edgecolor': 'black'})

# Add title
plt.title('Job Title Distribution (%)', fontsize=14, fontweight='bold', pad=15)

# Show chart
plt.show()


# In[119]:


# Select categorical columns
categorical_cols = ['Gender', 'Education Level']

# Iterate over each categorical feature
for col in categorical_cols:
    plt.figure(figsize=(10, 6))

    # Create barplot with color palette
    ax = sns.barplot(x=df[col].value_counts().index, 
                     y=df[col].value_counts().values, 
                     palette="coolwarm")

    # Set title and labels
    plt.title(f'{col} Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(col, fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, fontsize=10)

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # Add legend outside the chart
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in sns.color_palette("coolwarm", len(df[col].unique()))]
    plt.legend(handles, df[col].value_counts().index, title=col, loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
    plt.show()


# In[123]:


# Set Seaborn style for better visuals
sns.set_style("whitegrid")  

# Create histogram
plt.figure(figsize=(12, 7))
ax = sns.histplot(df['Salary'], kde=True, bins=30, color='royalblue', edgecolor='black', linewidth=1.2)

# Customize title and labels
plt.title('Salary Distribution - Histogram', fontsize=16, fontweight='bold', pad=15, color='darkblue')
plt.xlabel('Salary', fontsize=14, fontweight='bold', color='darkred')
plt.ylabel('Frequency', fontsize=14, fontweight='bold', color='darkred')

# Add mean & median lines for reference
mean_salary = df['Salary'].mean()
median_salary = df['Salary'].median()
plt.axvline(mean_salary, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_salary:,.0f}')
plt.axvline(median_salary, color='red', linestyle='-', linewidth=2, label=f'Median: {median_salary:,.0f}')

# Add annotations
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, edgecolor='black')

# Grid & border adjustments
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines

# Show the plot
plt.show()


# In[125]:


# Set style
sns.set_style("whitegrid")

# Scatter Plot - Salary vs. Age (Hue = Gender) with Trend Line
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Age'], y=df['Salary'], hue=df['Gender'], 
                palette='coolwarm', alpha=0.7, edgecolor='black', s=80)
sns.regplot(x=df['Age'], y=df['Salary'], scatter=False, color='black', ci=None)  # Trend Line

plt.title('Salary vs. Age (Grouped by Gender)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Age', fontsize=12, fontweight='bold')
plt.ylabel('Salary', fontsize=12, fontweight='bold')
plt.legend(title='Gender', loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Scatter Plot - Salary vs. Years of Experience (Hue = Education Level) with Trend Line
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Years of Experience'], y=df['Salary'], hue=df['Education Level'], 
                palette='viridis', alpha=0.7, edgecolor='black', s=80)
sns.regplot(x=df['Years of Experience'], y=df['Salary'], scatter=False, color='black', ci=None)  # Trend Line

plt.title('Salary vs. Years of Experience (Grouped by Education Level)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Years of Experience', fontsize=12, fontweight='bold')
plt.ylabel('Salary', fontsize=12, fontweight='bold')
plt.legend(title='Education Level', loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[198]:


# Create a pairplot for all numerical columns in the dataset
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50}, diag_kws={'shade': True})

# Show the plot
plt.show()


# In[151]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Education Level'], y=df['Salary'], palette="viridis", width=0.5, linewidth=2, notch=True)
sns.stripplot(x=df['Education Level'], y=df['Salary'], color='black', size=3, jitter=True, alpha=0.5)

plt.title('Salary Distribution by Education Level', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Education Level', fontsize=14, fontweight='bold')
plt.ylabel('Salary', fontsize=14, fontweight='bold')

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# In[ ]:





# In[156]:


import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.figure(figsize=(12, 6))

# Create violin plot
sns.violinplot(x=df['Education Level'], y=df['Salary'], palette="coolwarm", inner="box", linewidth=2)

# Title and labels
plt.title('Salary Distribution by Education Level (Violin Plot)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Education Level', fontsize=14, fontweight='bold')
plt.ylabel('Salary', fontsize=14, fontweight='bold')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=12)

# Add grid for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show plot
plt.show()


# In[160]:


import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set style
sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))

# Create violin plot with median line markers
sns.violinplot(x=df['Education Level'], y=df['Salary'], palette="coolwarm", inner="quartile", linewidth=2, alpha=0.8)

# Overlay swarm plot for individual points (avoiding overlap issues)
sns.swarmplot(x=df['Education Level'], y=df['Salary'], color='black', size=3, alpha=0.6)

# Title and labels
plt.title('Salary Distribution by Education Level (Violin Plot)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Education Level', fontsize=14, fontweight='bold')
plt.ylabel('Salary', fontsize=14, fontweight='bold')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=12)

# Add grid for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add legend manually to clarify colors
handles = [plt.Line2D([0], [0], color="black", marker='o', linestyle='', markersize=7, label="Individual Salaries"),
           plt.Line2D([0], [0], color="black", linestyle='-', linewidth=2, label="Quartiles")]
plt.legend(handles=handles, loc='upper left', fontsize=12, frameon=True, edgecolor="black")

# Show plot
plt.show()


# In[278]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Selecting relevant features
X = df[['Years of Experience']]  # Independent variable
y = df['Salary']  # Target variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Dictionary to store model results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Predictions
    
    # Compute residuals (Actual - Predicted)
    residuals = y_test - y_pred

    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "MSE": mse, "R² Score": r2, "Predictions": y_pred}

    # Scatter plot for Actual vs. Predicted Salaries
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label="Predictions", color="blue")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', label="Ideal Prediction")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title(f"Actual vs. Predicted Salary ({name})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Residual Analysis Plot
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_test, y=y_pred, lowess=True, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.xlabel("Actual Salary")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residuals Plot ({name})")
    plt.axhline(y=0, color="black", linestyle="--")  # Zero residual line
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T[['MAE', 'MSE', 'R² Score']]
print("\nModel Performance Comparison:\n")
print(results_df)

# Train the models (if not already trained)
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(X_train, y_train)

random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(X_train, y_train)

# Save models
joblib.dump(linear_regressor, "linear_regressor.pkl")
joblib.dump(decision_tree_regressor, "decision_tree_regressor.pkl")
joblib.dump(random_forest_regressor, "random_forest_regressor.pkl")

print("Models saved successfully!")



# In[208]:


# Sort values for a smoother line plot
X_test_sorted, y_test_sorted = zip(*sorted(zip(X_test.values.flatten(), y_test)))
X_test_sorted = np.array(X_test_sorted)

plt.figure(figsize=(12, 6))

# Plot actual salaries
sns.lineplot(x=X_test_sorted, y=y_test_sorted, marker='o', linestyle='-', label='Actual Salary', color='black')

# Plot predicted salaries for each model
for name, model in models.items():
    y_pred_sorted = model.predict(X_test_sorted.reshape(-1, 1))  # Predict using sorted X values
    
    # Change Decision Tree line color to dark red
    line_color = "darkred" if name == "Decision Tree Regression" else None  
    
    sns.lineplot(x=X_test_sorted, y=y_pred_sorted, marker='o', linestyle='--', label=f'Predicted ({name})', color=line_color)

# Add labels, title, and legend
plt.title("Actual vs. Predicted Salaries", fontsize=14, fontweight="bold")
plt.xlabel("Years of Experience", fontsize=12)
plt.ylabel("Salary", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()


# In[282]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models
linear_regressor = joblib.load("C:/Users/Grancy/salary_price_prediction/linear_regressor.pkl")
decision_tree_regressor = joblib.load("C:/Users/Grancy/salary_price_prediction/decision_tree_regressor.pkl")
random_forest_regressor = joblib.load("C:/Users/Grancy/salary_price_prediction/random_forest_regressor.pkl")

# Streamlit App
st.title("Employee Salary Prediction App")
st.sidebar.header("User Input")

# User Input
years_experience = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

if st.sidebar.button("Predict Salary"):
    input_features = np.array([[years_experience]])
    linear_pred = linear_regressor.predict(input_features)[0]
    decision_tree_pred = decision_tree_regressor.predict(input_features)[0]
    random_forest_pred = random_forest_regressor.predict(input_features)[0]

    # Display Predicted Salaries
    st.subheader("Predicted Salaries")
    results_df = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
        "Predicted Salary": [linear_pred, decision_tree_pred, random_forest_pred]
    })
    st.table(results_df)

    # Visualization
    st.subheader("Actual vs. Predicted Salaries")
    fig, ax = plt.subplots(figsize=(8, 6))
    y_test = np.linspace(30000, 150000, 50)  # Simulated actual salary values
    y_pred = np.linspace(30000, 150000, 50)  # Simulated predicted salary values
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label="Predictions", color="blue", ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label="Ideal Prediction")
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Actual vs. Predicted Salary")
    ax.legend()
    st.pyplot(fig)

    # Residuals Plot
    st.subheader("Residuals Plot (Linear Regression)")
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = y_test - y_pred
    sns.residplot(x=y_test, y=y_pred, lowess=True, scatter_kws={"color": "blue"}, line_kws={"color": "red"}, ax=ax)
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residuals Plot (Linear Regression)")
    ax.axhline(y=0, color="black", linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig)

    st.success("Prediction complete!")

