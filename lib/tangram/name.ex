defmodule AkinML.Tangram.Name do
  @moduledoc """
  1. [Install](https://www.tangram.dev/docs/install) the Tangram CLI
  1. Training data is in `names.csv` in same directory as this module
  1. Train the model: `tangram train --file names.csv --target outcome`
  1. Model is `metrics_for_training.tangram`
  1. Run the Tangram app: `tangram app`
  1. Access at http://localhost:8080/ to interact with the model, features, predictions, etc.
  """

  @doc """
  Run the predictions using the model
  """
  def run() do
    model_path = "metrics_for_training.tangram"
    IO.inspect model_path

    # Load the model
    model = Tangram.load_model_from_path(model_path)

    # Build inputs from data ready for predictions
    File.stream!("tangram_predictions.csv")
    |> Stream.map(&String.trim(&1))
    |> Enum.to_list()
    |> Enum.each(fn row ->
      [bag_distance, chunk_set, dice_sorensen, metaphone, double_metaphone,
      double_metaphone_chunks, jaccard, jaro_winkler, levenshtein, ngram, overlap,
      sorted_chunks, tversky, name, _match] = String.split(row, "\t")
      input = %{
        :bag_distance => bag_distance,
        :chunk_set => chunk_set,
        :dice_sorensen => dice_sorensen,
        :metaphone => metaphone,
        :double_metaphone => double_metaphone,
        :double_metaphone_chunks => double_metaphone_chunks,
        :jaccard => jaccard,
        :jaro_winkler => jaro_winkler,
        :levenshtein => levenshtein,
        :ngram => ngram,
        :overlap => overlap,
        :sorted_chunks => sorted_chunks,
        :tversky => tversky
      }

      # Make the prediction!
      output = Tangram.predict(model, input)
      # id = make_ref()
      #   |> :erlang.ref_to_list()
      #   |> List.to_string()
      # indentifier = "#{id}: #{name}"
      indentifier = "#{name}"
      log_predictions = %Tangram.LogPredictionArgs{
        :identifier => indentifier,
        :input => input,
        :options => nil,
        :output => output
      }
      Tangram.log_prediction(model, log_predictions)

      # Print the output.
      IO.write("Prediction Identifier: ")
      IO.inspect(indentifier)
      IO.write("Output: ")
      IO.inspect(output)
    end)
  end

  @doc """
  Log true values
  """
  def truth() do
    model_path = "metrics_for_training.tangram"

    # Load the model
    model = Tangram.load_model_from_path(model_path)

    File.stream!("tangram_predictions.csv")
    |> Stream.map(&String.trim(&1))
    |> Enum.to_list()
    |> Enum.each(fn row ->
      [_, _, _, _, _, _, _, _, _, _, _, _, _, name, match] = String.split(row, "\t")

      indentifier = "#{name}"
      true_value = %Tangram.LogTrueValueArgs{
        :identifier => indentifier,
        :true_value => to_string(match),
      }
      Tangram.log_true_value(model, true_value)
    end)
  end

  @doc """
  Run the predictions using the model
  """
  def predict(input, names) do
    model_path = "metrics_for_training.tangram"

    # Load the model
    model = Tangram.load_model_from_path(model_path)

    # Make the prediction!
    output = Tangram.predict(model, input)
    # id = make_ref()
    #   |> :erlang.ref_to_list()
    #   |> List.to_string()
    # indentifier = "#{id}: #{name}"
    indentifier = "#{names}"
    log_predictions = %Tangram.LogPredictionArgs{
      :identifier => indentifier,
      :input => input,
      :options => nil,
      :output => output
    }
    Tangram.log_prediction(model, log_predictions)

    # Print the output.
    IO.write("Prediction Identifier: ")
    IO.inspect(indentifier)
    IO.write("Output: ")
    IO.inspect(output)
  end
end
