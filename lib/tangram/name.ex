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
  def predict() do
    model_path = "lib/tangram/metrics_for_training.tangram"

    # Load the model
    model = Tangram.load_model_from_path(model_path)

    # Build inputs from data ready for predictions
    # File.stream!("lib/tangram/metrics_for_predicting.csv")
    File.stream!("lib/tangram/mini_metrics_for_predicting.csv")
    |> Stream.map(&String.trim(&1))
    |> Enum.to_list()
    |> Enum.each(fn row ->
      [bag_distance, substring_set, dice_sorensen, metaphone, double_metaphone,
      substring_double_metaphone, jaccard, jaro_winkler, levenshtein, ngram, overlap,
      substring_sort, tversky, initials, name, _match] = String.split(row, "\t")
      input = %{
        :bag_distance => bag_distance,
        :substring_set => substring_set,
        :dice_sorensen => dice_sorensen,
        :metaphone => metaphone,
        :double_metaphone => double_metaphone,
        :substring_double_metaphone => substring_double_metaphone,
        :jaccard => jaccard,
        :jaro_winkler => jaro_winkler,
        :levenshtein => levenshtein,
        :ngram => ngram,
        :overlap => overlap,
        :substring_sort => substring_sort,
        :tversky => tversky,
        :initials => initials
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
    model_path = "lib/tangram/metrics_for_training.tangram"

    # Load the model
    model = Tangram.load_model_from_path(model_path)

    File.stream!("lib/tangram/mini_metrics_for_predicting.csv")
    |> Stream.map(&String.trim(&1))
    |> Enum.to_list()
    |> Enum.each(fn row ->
      [_, _, _, _, _, _, _, _, _, _, _, _, _, _, name, match] = String.split(row, "\t")

      indentifier = "#{name}"
      true_value = %Tangram.LogTrueValueArgs{
        :identifier => indentifier,
        :true_value => to_string(match),
      }
      Tangram.log_true_value(model, true_value)
    end)
  end

  @spec predict(binary(), list() | map(), binary()) :: list() | %Tangram.BinaryClassificationPredictOutput{}
  @doc """
  Run the predictions using the model
  """
  def predict(model_path, inputs, identifier) when is_list(inputs) do
    Enum.map(inputs, fn input ->
      predict(model_path, input, identifier)
    end)
  end

  def predict(model_path, input, identifier) do
    model = Tangram.load_model_from_path(model_path)
    Tangram.predict(model, input)
    |> log_prediction(model, input, identifier)
  end

  defp log_prediction(output, model, input, identifier) do
    log = %Tangram.LogPredictionArgs{
      :identifier => "#{identifier}",
      :input => input,
      :options => nil,
      :output => output
    }
    Tangram.log_prediction(model, log)
    output
  end
end
