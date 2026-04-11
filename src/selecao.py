def columns_filter(dataframe, selected_columns):
    columns = selected_columns
    return dataframe[columns]

def filter_by_campus(df, campus_input):
    return df[df['CAMPUS CENTRO DO CURSO']==campus_input.upper()]

def filter_by_und_acad(df, instituo_input):
    return df[df['CENTRO DO CURSO SIGLA']==instituo_input.upper()]

def filter_by_curso(df, curso_input):
    return df[df['CURSO']==curso_input.upper()]

def filter_dataframe(df):
    print('Selecione um campus:\n')

    campus_op = df['CAMPUS CENTRO DO CURSO'].unique().tolist()
    campus_op.sort()

    for campus in campus_op:
        print(campus)

    campus_input=input()
    
    df_campus = filter_by_campus(df, campus_input)
    
    df_institutos = df_campus[['CENTRO DO CURSO', 'CENTRO DO CURSO SIGLA']]
    df_institutos = df_institutos.drop_duplicates()
    df_institutos = df_institutos.sort_values(by='CENTRO DO CURSO')

    print('\n\n\nSelecione a Unidade Acadêmica pela sigla:\n')
    for und_acad in df_institutos.itertuples():
        print(f'{und_acad[2]} - {und_acad[1]}')

    und_input = input()
    
    df_und_acad = filter_by_und_acad(df_campus, und_input)
    
    print(df_und_acad)
    
    cursos = df_und_acad['CURSO'].unique().tolist()
    cursos.sort()    

    print('\n\n\nSelecione o Curso:\n')
    for curso in cursos:
        print(curso)
        
    curso_input= input()
    
    df_curso = filter_by_curso(df_und_acad, curso_input)
    
    return df_curso

def select_label(df, label:int):
    return df[df['label']==label]