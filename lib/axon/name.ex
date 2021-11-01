defmodule AkinML.Axon.Name do
  require Axon
  require Logger

  @epochs 10
  @learning_rate 0.001
  @loss :mean_squared_error
  @dropout_rate 0.1

  @input_columns [
    "Bag Distance",
    "Chunk Set",
    "Dice Sorensen",
    "Metaphone",
    "Double Metaphone",
    "Double Metaphone Chunks",
    "Jaccard",
    "Jaro-Winkler",
    "Levenshtein",
    "NGram",
    "Overlap",
    "Sorted Chunks",
    "Tversky",
    "Initials",
    "Match"
  ]

  def train() do
    # split the data into test and train sets, each with inputs and targets
    {test_inputs, test_targets, train_inputs, train_targets} =
      Axon.Data.split_inputs(
        "./lib/axon/new.data",
        input_columns: @input_columns,
        target_column: "Match",
        test_train_ratio: 0.1
      )

    # train the model
    model = do_training(train_inputs, train_targets)

    # make some predictions
    test_inputs
    |> Enum.zip(test_targets)
    |> Enum.each(fn {name_input, actual_match} ->
      predicted_match = predict(model, name_input)
        |> Float.round(8)
      actual_match = scalar(actual_match)
      color = case actual_match do
        0 ->
          if 0.98 > predicted_match, do: :green, else: :red
        1 ->
          if predicted_match > 0.98 do
            :green
          else
            :red
          end
      end
      Logger.info("Actual: #{actual_match}. Predicted: #{predicted_match}.", ansi_color: color)
    end)
  end

  def do_training(inputs, targets) do
    model =
      Axon.input({nil, Enum.count(@input_columns)})
      |> Axon.dense(14)
      |> Axon.dropout(rate: @dropout_rate)
      |> Axon.dense(1)

    optimizer = Axon.Optimizers.adamw(@learning_rate)

    %{params: trained_params} =
      model
      |> Axon.Training.step(@loss, optimizer)
      |> Axon.Training.train(inputs, targets, epochs: @epochs)

    {model, trained_params}
  end

  def predict({model, trained_params}, name_input) do
    model
    |> Axon.predict(trained_params, name_input)
    |> Nx.to_flat_list()
    |> List.first()
  end

  def scalar(tensor) do
    tensor |> Nx.to_flat_list() |> List.first()
  end
end
