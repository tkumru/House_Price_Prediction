from pyspark.sql import functions as F

def replace_null(df):
    columns = df.columns
    
    for col in columns:
        df = df.withColumn(col,
                           F.when(F.col(col) == "", None).otherwise(
                           F.when(F.col(col) == "NA", None).otherwise(
                           F.when(F.col(col).isNull(), None).otherwise(
                           F.col(col)))))
    
    return df

########################################################
# Delete
########################################################

def remove_missing_values(df, columns: list):
    return df.dropna(how="any", subset=columns)

########################################################
# Fill Categorical Columns with Mode
########################################################

def fill_cat_with_mode(df, columns: list):
    for col in columns:
        mode = df.select(col).toPandas()[col].mode()[0]
        
        df = df.na.fill(value=mode, subset=[col])
        
    return df

########################################################
# Fill Numerical Columns with Mode
########################################################

def fill_num_with_mode(df, columns: list):
    for col in columns:
        mode = df.select(F.percentile_approx(col, 0.5)).first()[0]
        
        df = df.na.fill(value=mode, subset=[col])
        
    return df
