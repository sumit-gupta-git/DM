import click, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

@click.command()
@click.argument("filepath")
@click.argument("target")
@click.option("--criterion", type=click.Choice(["gini", "entropy", "log_loss", "auto"], case_sensitive=False),
              default="auto", help="Split criterion (default: auto chooses best by accuracy).")
@click.option("--max_depth", default=None, type=int, help="Maximum depth of the tree (default: None).")
@click.option("--test_size", default=0.2, help="Proportion of data for testing.")
@click.option("--save_img", is_flag=True, help="Save decision tree image in /data directory.")
def decision_tree(filepath, target, criterion, max_depth, test_size, save_img):
    """Smart Decision Tree Classifier with auto criterion selection and optional image saving."""

    # --- Load and path validation ---
    if not os.path.exists(filepath):
        click.echo(f" File not found: {filepath}")
        return

    #  delimiter
    with open(filepath) as f:
        sep = ';' if f.readline().count(';') > 1 else ','
    df = pd.read_csv(filepath, sep=sep).dropna()
    click.echo(f" Loaded: {filepath} ({len(df)} rows, sep='{sep}')")

    if target not in df.columns:
        click.echo(f" Target '{target}' not found. Columns: {list(df.columns)}")
        return

    # --- Auto-bin numeric target ---
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
        click.echo(f"'{target}' seems continuous — converting to Low/Med/High bands.")
        df[target + "_cat"] = pd.cut(df[target], bins=[-1, 9, 12, 20],
                                    labels=["Low", "Medium", "High"])
        target = target + "_cat"

    # --- Encode categoricals ---
    for col in df.select_dtypes(include='object').columns:
        if col != target:
            df[col] = LabelEncoder().fit_transform(df[col])

    X, y = df.drop(columns=[target]), df[target]

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # --- Scale ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Auto selection ---
    criteria = ["gini", "entropy", "log_loss"]
    results = {}

    if criterion == "auto":
        click.echo(" Auto mode: testing all criteria...\n")
        for crit in criteria:
            clf = DecisionTreeClassifier(criterion=crit, max_depth=max_depth, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[crit] = acc
            click.echo(f"   • {crit:<10} → Accuracy: {acc:.4f}")
        criterion = max(results, key=results.get)
        click.echo(f"\n Best criterion: {criterion} (Accuracy: {results[criterion]:.4f})\n")

    # --- Train Final Model ---
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    click.echo(f"Final Model: DecisionTree ({criterion}) | Accuracy: {acc:.4f}\n")
    click.echo(classification_report(y_test, y_pred, zero_division=0))

    # --- Visualization ---
    click.echo("Generating tree visualization...")
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=[str(c) for c in clf.classes_],
              filled=True, rounded=True, fontsize=8)
    plt.title(f"Decision Tree ({criterion})", fontsize=14)

    # Save only if requested
    if save_img:
        img_dir = "data"
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, "decision_tree.png")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        click.echo(f" Image saved at: {img_path}")

    # Always preview
    plt.show()
    plt.close()

if __name__ == "__main__":
    decision_tree()
