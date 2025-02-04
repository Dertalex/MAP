import os
import sys
os.system("conda init bash")
os.system("conda activate proteusAI")
import proteusAI as pai

print(os.getcwd())
sys.path.append("src/")


# will initiate storage space - else in memory
datasets = ["data/NOD_AT_edit.csv"]
y_columns = ["Data"]

results_dictionary = {}
for dataset in datasets:
    for y in y_columns:
        # load data from csv or excel: x should be sequences, y should be labels, y_type class or num
        library = pai.Library(
            source=dataset,
            seqs_col="Sequence",
            y_col=y,
            y_type="num",
            names_col="Description",
        )

        # compute and save ESM-2 representations at example_lib/representations/esm2
        library.compute(method="ohe", batch_size=10)

        # define a model
        model = pai.Model(library=library, k_folds=5, model_type="rf", x="blosum62")

        # train model
        model.train()

        # test prediction capabilities on old data
        out_pred = model.val_predictions
        print(out_pred)
        # search for new mutants
        # out_search = model.search(optim_problem="max")

        # save results
        # outpath = "test/results/"
        # if not os.path.exists(outpath):
        #     os.makedirs(outpath, exist_ok=True)
        #
        # out_search.to_csv(os.path.join(outpath, "results.csv"))
