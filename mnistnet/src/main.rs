extern crate csv;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate serde;
extern crate serde_json;

mod network;

use indicatif::{ProgressBar, ProgressStyle};
use std::fs;

/// Load a set of inputs and outputs (rust-csv can probably do this directly)
fn load_data(
    inputs_file: &str,
    labels_file: &str,
    expected: u32,
) -> Vec<(network::Input, network::Label)> {
    let mut inputs_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(inputs_file)
        .unwrap();
    let mut labels_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(labels_file)
        .unwrap();

    let pb = ProgressBar::new(expected as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] {wide_bar} ({eta})"));

    let mut data = Vec::new();
    for (input, label) in inputs_reader.deserialize().zip(labels_reader.deserialize()) {
        let input: Vec<f64> = input.unwrap();
        let label: u32 = label.unwrap();

        // Convert into inputs/labels
        let input = network::vector_to_input(input);
        let label = network::Label::from_num(label);

        data.push((input, label));

        pb.inc(1);
    }
    pb.finish_with_message("Done loading");

    assert!(data.len() == expected as usize);

    data
}

fn main() {
    println!("Loading the training data");
    let training_data = load_data("data/TrainDigitX.csv", "data/TrainDigitY.csv", 50000);
    println!("Loading the testing data");
    let testing_data = load_data("data/TestDigitX.csv", "data/TestDigitY.csv", 10000);
    println!("Done loading data");

    let learning_rates = vec![0.1, 0.5, 1., 3.];
    let hidden_sizes = vec![16, 32, 64, 128, 256];

    for lr in &learning_rates {
        for hs in &hidden_sizes {
            let mut network = network::Network::new(vec![784, *hs, 10]);
            network.train(training_data.clone(), 60, 10, *lr, testing_data.clone());

            // Save the trained network
            fs::write(
                format!("saved_networks/{}_{}.net", lr, hs),
                serde_json::to_string(&network)
                    .expect("An error occured while serializing a network"),
            )
            .expect("An error occured while writing a saved model to disk");
        }
    }
}
