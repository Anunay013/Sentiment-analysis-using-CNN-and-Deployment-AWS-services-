"""
Main sentiment training script

Author pharnoux


"""

import os
import argparse
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# import model_training.sentiment_dataset as sentiment_dataset
# import model_training.sentiment_model_cnn as sentiment_model_cnn
# import model_training.config_holder as config_holder
import sentiment_dataset
import sentiment_model_cnn
import config_holder

def main(args):
    """
    Main training method

    """

    print("Preparing for training...")

    training_config = config_holder.ConfigHolder(args.config_file).config

    training_config["num_epoch"] = args.num_epoch

    train_dataset = sentiment_dataset.train_input_fn(args.train, training_config)
    validation_dataset = sentiment_dataset.validation_input_fn(args.validation, training_config)
    eval_dataset = sentiment_dataset.eval_input_fn(args.eval, training_config)

    model = sentiment_model_cnn.keras_model_fn(None, training_config)

    print("Starting training...")

    model.fit(
        x=train_dataset[0], y=train_dataset[1], steps_per_epoch=train_dataset[2]["num_batches"],
        epochs=training_config["num_epoch"],
        validation_data=(validation_dataset[0], validation_dataset[1]),
        validation_steps=validation_dataset[2]["num_batches"])

    score = model.evaluate(
        eval_dataset[0], eval_dataset[1], steps=eval_dataset[2]["num_batches"], verbose=0)

    print("Test loss:{}".format(score[0]))
    print("Test accuracy:{}".format(score[1]))

    sentiment_model_cnn.save_model(model, os.path.join(args.model_output_dir, "sentiment_model.h5"))

def get_arg_parser():
    """
    Adding this method to unit test

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        help="The directory where the training data is stored.")
    parser.add_argument(
        "--validation",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_VALIDATION"),
        help="The directory where the validation data is stored.")
    parser.add_argument(
        "--eval",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_EVAL"),
        help="The directory where the evalutaion data is stored.")
    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=False,
        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10,
        help="The number of steps to use for training.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_config.json"),
        help="The path to the training config file.")

    return parser


if __name__ == "__main__":
    PARSER = get_arg_parser()
    ARGS = PARSER.parse_args()
    main(ARGS)
