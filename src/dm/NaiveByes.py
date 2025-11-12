import click, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

@click.command()
@click.argument("filepath")
@click.argument("target")
@click.option("--model", type=click.Choice(["auto","gaussian","multinomial","bernoulli"], case_sensitive=False),
              default="auto", help="Naive Bayes model type (default: auto chooses best by accuracy).")
@click.option("--test_size", default=0.2, help="Proportion of data for testing.")
def naive_bayes(filepath, target, model, test_size):
    """Smart Naive Bayes classifier with auto model selection."""
    if not os.path.exists(filepath):
        click.echo(f" File not found: {filepath}")
        return

    # Detect delimiter and load
    with open(filepath) as f:
        sep = ';' if f.readline().count(';') > 1 else ','
    df = pd.read_csv(filepath, sep=sep).dropna()
    click.echo(f"Loaded: {filepath} ({len(df)} rows, sep='{sep}')")

    if target not in df.columns:
        click.echo(f"Target '{target}' not found. Columns: {list(df.columns)}")
        return

    # Auto-bin 
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
        click.echo(f"'{target}' seems continuous — converting to Low/Med/High bands.")
        df[target+"_cat"] = pd.cut(df[target], bins=[-1,9,12,20],
                                   labels=["Low","Medium","High"])
        target = target+"_cat"

    # Encode non-target categoricals
    for col in df.select_dtypes(include='object').columns:
        if col != target:
            df[col] = LabelEncoder().fit_transform(df[col])

    X, y = df.drop(columns=[target]), df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # models 
    models = {
        "gaussian": (GaussianNB(), StandardScaler()),
        "multinomial": (MultinomialNB(), MinMaxScaler()),
        "bernoulli": (BernoulliNB(), MinMaxScaler())
    }

    results = {}

    # Auto: train all 3 and compare accuracie
    if model == "auto":
        click.echo("Auto mode: testing all models for best accuracy...\n")
        for name, (clf, scaler) in models.items():
            X_scaled = scaler.fit_transform(X_train)
            X_t = scaler.transform(X_test)
            clf.fit(X_scaled, y_train)
            y_pred = clf.predict(X_t)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            click.echo(f"   • {name.title():<11} → Accuracy: {acc:.4f}")

        # best
        model = max(results, key=results.get)
        click.echo(f"\n Best model selected: {model.title()} NB (Accuracy: {results[model]:.4f})\n")

    # Use best  model
    clf, scaler = models[model]
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    click.echo(f"Final Model: {model.title()} NB | Accuracy: {acc:.4f}\n")
    click.echo(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    naive_bayes()
