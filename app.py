Traceback:
File "/mount/src/streamlit-lbw-app/app.py", line 147, in <module>
    prob = float(model.predict_proba(X)[0, 1])
                 ~~~~~~~~~~~~~~~~~~~^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py", line 1921, in predict_proba
    class_probs = super().predict(
        X=X,
    ...<2 lines>...
        iteration_range=iteration_range,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 774, in inner_f
    return func(**kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py", line 1446, in predict
    predts = self.get_booster().inplace_predict(
        data=X,
    ...<4 lines>...
        validate_features=validate_features,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 774, in inner_f
    return func(**kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 2887, in inplace_predict
    _check_call(
    ~~~~~~~~~~~^
        _LIB.XGBoosterPredictFromColumnar(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<7 lines>...
        )
        ^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 323, in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
