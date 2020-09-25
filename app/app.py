from flask import Flask, render_template, request, send_from_directory
import joblib
import os
import werkzeug
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from numpy import nan



app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def predicts():
    #送信されたら以下のif文がTrue
    if request.method == 'POST':
        f = request.files.get('csv')
        filepath = 'static/csv/' + secure_filename(f.filename)
        f.save(filepath)
        test = pd.read_csv(filepath)
        # 変数に格納したらすてる
        os.remove(filepath)

        test['拠点番号'] = test['拠点番号'].astype('category').cat.codes
        # # testデータからデータの取り出し
        test_data = test.loc[:,['フラグオプション選択', '派遣形態', '紹介予定派遣', '正社員登用あり', '給与/交通費　給与下限','学校・公的機関（官公庁）', '経験者優遇','会社概要　業界コード','車通勤OK','拠点番号']].values

        # # 提出データの予測
        gs = joblib.load('./predict.pkl')
        y_predd = gs.predict(test_data)
        predict_file = pd.DataFrame({'お仕事No.' : test['お仕事No.'], '応募数 合計' : y_predd})
        predict_file.to_csv('predict.csv', index = False)

        return render_template('result.html')

    else:
        return render_template('index.html')


@app.route('/download', methods=['GET','POST'])
def export_action():
    return send_from_directory(
        directory='./',
        filename='predict.csv',
        as_attachment=True,
        attachment_filename='predict.csv',
    )



if __name__ == "__main__":
    app.run(port=8000, debug=True)