from project.submission_script import load_test_data, make_predictions
def test_eval_not_throw_errors():
    test_data = load_test_data('src/project/data/')
    predictions = make_predictions(test_data)
