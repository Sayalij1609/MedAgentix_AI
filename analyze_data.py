import pandas as pd, os, sys
sys.stdout.reconfigure(encoding='utf-8')

d = 'datasets/new_data'

# 1. DiseaseAndSymptoms
df = pd.read_csv(os.path.join(d, 'DiseaseAndSymptoms.csv'))
print(f"DiseaseAndSymptoms: {df.shape[0]} rows, {df['Disease'].nunique()} diseases")
print(f"  Sample diseases: {sorted(df['Disease'].unique())[:15]}")
print()

# 2. Final_Augmented (largest training set)
df2 = pd.read_csv(os.path.join(d, 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'), usecols=['diseases'])
print(f"Final_Augmented: {df2.shape[0]} rows, {df2['diseases'].nunique()} diseases")
print(f"  Sample diseases: {sorted(df2['diseases'].unique())[:15]}")
print()

# 3. Diseases_Symptoms
df3 = pd.read_csv(os.path.join(d, 'Diseases_Symptoms.csv'))
print(f"Diseases_Symptoms: {df3.shape[0]} rows, {df3['Name'].nunique()} diseases")
print()

# 4. Healthcare SymptomDiseaseDrug
df4 = pd.read_csv(os.path.join(d, 'Healthcare SymptomDiseaseDrug Research Dataset.csv'))
print(f"HealthcareSymptomDiseaseDrug: {df4.shape[0]} rows, {df4['disease'].nunique()} diseases")
print(f"  Extra cols: {[c for c in df4.columns if c not in ['case_id','symptom_1','symptom_2','symptom_3','symptom_4','symptom_5','disease']]}")
print()

# 5-9. Knowledge bases
for name, fname, col in [
    ('Description', 'description.csv', 'Disease'),
    ('Medications', 'medications.csv', 'Disease'),
    ('Precautions', 'precautions.csv', 'Disease'),
    ('Diets', 'diets.csv', 'Disease'),
    ('Workout', 'workout.csv', 'Disease'),
    ('Symptom-severity', 'Symptom-severity.csv', 'Symptom'),
]:
    df5 = pd.read_csv(os.path.join(d, fname))
    print(f"{name}: {df5.shape[0]} entries, columns={list(df5.columns)}")

print()
# 10. medical_question_answer
df10 = pd.read_csv(os.path.join(d, 'medical_question_answer_dataset_50000.csv'), nrows=3)
print(f"MedQA 50k columns: {list(df10.columns)}")
print(f"  Sample: {df10.iloc[0].to_dict()}")
