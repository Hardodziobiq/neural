mod cli;

use std::{env, error::Error};

use cli::{Command, Options};

use neural::net::{
    activation_function::ActivationFunction,
    data_set::{Dataset, load_classification, load_regression},
    loss::Loss,
    neural_net::NeuralNet,
};

type AppResult<T> = Result<T, Box<dyn Error>>;

fn main() -> AppResult<()> {
    match cli::parse(&env::args().skip(1).collect::<Vec<_>>())? {
        Command::Demo => {
            regression(Options::demo("regression"))?;
            classification(Options::demo("classification"), 3)
        }
        Command::Help => {
            println!("{}", cli::HELP);
            Ok(())
        }
        Command::Regression(options) => regression(options),
        Command::Classification(options, classes) => classification(options, classes),
    }
}

fn regression(options: Options) -> AppResult<()> {
    let mut train = load_regression(&options.train_path)?;
    let mut test = load_regression(&options.test_path)?;
    let inputs = train.features.first().map_or(0, Vec::len);
    train.validate(inputs, 1)?;
    test.validate(inputs, 1)?;
    let sample = test.features[0].clone();
    let features = train.standardize_features()?;
    features.transform(&mut test.features)?;
    let targets = train.standardize_targets()?;
    targets.transform(&mut test.targets)?;

    let mut net = NeuralNet::new(
        42,
        inputs,
        true,
        vec![16, 16, 1],
        ActivationFunction::Tanh,
        ActivationFunction::Linear,
        Loss::MeanSquaredError,
        0.03,
    )?;
    let history = net.train(&train, Some(&test), options.epochs, options.batch_size)?;
    if let Some(path) = options.plot {
        history.save_loss_plot(&path)?;
        println!("Loss plot saved to {}", path.display());
    }

    println!("Regression");
    println!("Training MSE (standardized): {:.6}", net.test(&train)?);
    println!("Test MSE (standardized): {:.6}", net.test(&test)?);
    let input = features.transform_sample(&sample)?;
    let prediction = targets.inverse_transform_sample(&net.predict(&input)?)?;
    println!("Prediction for {sample:?}: {:.6}\n", prediction[0]);
    Ok(())
}

fn classification(options: Options, classes: usize) -> AppResult<()> {
    let mut train = classification_dataset(&options.train_path)?;
    let mut test = classification_dataset(&options.test_path)?;
    let inputs = train.features.first().map_or(0, Vec::len);
    train.validate(inputs, 1)?;
    test.validate(inputs, 1)?;
    train.one_hot_encode_targets(classes)?;
    test.one_hot_encode_targets(classes)?;
    let features = train.standardize_features()?;
    features.transform(&mut test.features)?;

    let mut net = NeuralNet::new(
        42,
        inputs,
        true,
        vec![8, classes],
        ActivationFunction::Tanh,
        ActivationFunction::SoftMax,
        Loss::CategorialCrossEntropy,
        0.05,
    )?;
    let history = net.train(&train, Some(&test), options.epochs, options.batch_size)?;
    if let Some(path) = options.plot {
        history.save_loss_plot(&path)?;
        println!("Loss plot saved to {}", path.display());
    }

    let mut correct = 0;
    for (input, target) in test.features.iter().zip(&test.targets) {
        let probabilities = net.predict(input)?;
        let predicted = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .ok_or("No class probabilities")?
            .0;
        correct += usize::from(target[predicted] == 1.0);
    }
    println!("Classification");
    println!("Test cross-entropy: {:.6}", net.test(&test)?);
    println!(
        "Test accuracy: {:.2}%",
        100.0 * correct as f64 / test.features.len() as f64
    );
    Ok(())
}

fn classification_dataset(path: &str) -> AppResult<Dataset> {
    let (features, labels) = load_classification(path)?;
    Ok(Dataset {
        features,
        targets: labels.into_iter().map(|label| vec![label as f64]).collect(),
    })
}
