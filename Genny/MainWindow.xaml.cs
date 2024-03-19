using Genny.Utils;
using Genny.ViewModel;
using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace Genny
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private Model _model;
        private Tokenizer _tokenizer;

        private string _prompt;
        private string _modelPath = "D:\\Repositories\\phi2_onnx";
        private bool _isModelLoaded;
        private ResultModel _currentResult;
        private SearchOptionsModel _searchOptions;
        private CancellationTokenSource _cancellationTokenSource;

        public MainWindow()
        {
            ClearCommand = new RelayCommand(ClearAsync);
            CancelCommand = new RelayCommand(CancelAsync);
            OpenModelCommand = new RelayCommand(OpenModelAsync);
            LoadModelCommand = new RelayCommand(LoadModelAsync, CanExecuteLoadModel);
            SendPromptCommand = new RelayCommand(SendPromptAsync, CanExecuteSendPrompt);
            ResultHistory = new ObservableCollection<ResultModel>();
            SearchOptions = new SearchOptionsModel();
            InitializeComponent();
        }

        public RelayCommand ClearCommand { get; }
        public RelayCommand CancelCommand { get; }
        public RelayCommand OpenModelCommand { get; }
        public RelayCommand LoadModelCommand { get; }
        public RelayCommand SendPromptCommand { get; }
        public ObservableCollection<ResultModel> ResultHistory { get; }

        public bool IsModelLoaded
        {
            get { return _isModelLoaded; }
            set { _isModelLoaded = value; NotifyPropertyChanged(); }
        }

        public string ModelPath
        {
            get { return _modelPath; }
            set { _modelPath = value; NotifyPropertyChanged(); }
        }

        public string Prompt
        {
            get { return _prompt; }
            set { _prompt = value; NotifyPropertyChanged(); }
        }

        public SearchOptionsModel SearchOptions
        {
            get { return _searchOptions; }
            set { _searchOptions = value; NotifyPropertyChanged(); }
        }

        public ResultModel CurrentResult
        {
            get { return _currentResult; }
            set { _currentResult = value; NotifyPropertyChanged(); }
        }


        private Task OpenModelAsync()
        {
            var folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Model Folder Path",
                UseDescriptionForTitle = true,
            };
            var dialogResult = folderBrowserDialog.ShowDialog();
            if (dialogResult == System.Windows.Forms.DialogResult.OK)
                ModelPath = folderBrowserDialog.SelectedPath;

            return Task.CompletedTask;
        }


        private Task ClearAsync()
        {
            ResultHistory.Clear();
            return Task.CompletedTask;
        }


        private Task CancelAsync()
        {
            _cancellationTokenSource?.Cancel();
            return Task.CompletedTask;
        }


        private async Task LoadModelAsync()
        {
            await UnloadModelAsync();
            try
            {
                await Task.Run(() =>
                {
                    _model = new Model(ModelPath);
                    _tokenizer = new Tokenizer(_model);
                });
                IsModelLoaded = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Model Load Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        private bool CanExecuteLoadModel()
        {
            return !string.IsNullOrWhiteSpace(ModelPath);
        }


        private Task UnloadModelAsync()
        {
            _model?.Dispose();
            _tokenizer?.Dispose();
            IsModelLoaded = false;
            return Task.CompletedTask;
        }


        private async Task SendPromptAsync()
        {
            try
            {
                var userInput = new ResultModel
                {
                    Content = Prompt,
                    IsUserInput = true
                };

                Prompt = null;
                CurrentResult = null;
                ResultHistory.Add(userInput);
                _cancellationTokenSource = new CancellationTokenSource();
                await foreach (var sentencePiece in RunInferenceAsync(userInput.Content, _cancellationTokenSource.Token))
                {
                    if (CurrentResult == null)
                    {
                        if (string.IsNullOrWhiteSpace(sentencePiece.Content)) // Ingore preceding '\n'
                            continue;

                        ResultHistory.Add(CurrentResult = new ResultModel());
                    }
                    CurrentResult.Content += sentencePiece.Content;
                }
            }
            catch (OperationCanceledException)
            {
                CurrentResult.Content += "\n\n[Operation Canceled]";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Inference Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        private bool CanExecuteSendPrompt()
        {
            return !string.IsNullOrWhiteSpace(Prompt);
        }


        private async IAsyncEnumerable<TokenModel> RunInferenceAsync(string prompt, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var sequences = await Task.Run(() => _tokenizer.Encode(prompt), cancellationToken);

            using var generatorParams = new GeneratorParams(_model);
            ApplySearchOptions(generatorParams, SearchOptions);
            generatorParams.SetInputSequences(sequences);

            using var tokenizerStream = _tokenizer.CreateStream();
            using var generator = new Generator(_model, generatorParams);
            while (!generator.IsDone())
            {
                cancellationToken.ThrowIfCancellationRequested();

                yield return await Task.Run(() =>
                {
                    generator.ComputeLogits();
                    generator.GenerateNextTokenTop();

                    var tokenId = generator.GetSequence(0)[^1];
                    return new TokenModel(tokenId, tokenizerStream.Decode(tokenId));
                }, cancellationToken);
            }
        }


        public void ApplySearchOptions(GeneratorParams generatorParams, SearchOptionsModel searchOptions)
        {
            generatorParams.SetSearchOption("top_p", searchOptions.TopP);
            generatorParams.SetSearchOption("top_k", searchOptions.TopK);
            generatorParams.SetSearchOption("temperature", searchOptions.Temperature);
            generatorParams.SetSearchOption("repetition_penalty", searchOptions.RepetitionPenalty);
            generatorParams.SetSearchOption("past_present_share_buffer", searchOptions.PastPresentShareBuffer);
            generatorParams.SetSearchOption("num_return_sequences", searchOptions.NumReturnSequences);
            generatorParams.SetSearchOption("no_repeat_ngram_size", searchOptions.NoRepeatNgramSize);
            generatorParams.SetSearchOption("min_length", searchOptions.MinLength);
            generatorParams.SetSearchOption("max_length", searchOptions.MaxLength);
            generatorParams.SetSearchOption("length_penalty", searchOptions.LengthPenalty);
            generatorParams.SetSearchOption("early_stopping", searchOptions.EarlyStopping);
            generatorParams.SetSearchOption("do_sample", searchOptions.DoSample);
            generatorParams.SetSearchOption("diversity_penalty", searchOptions.DiversityPenalty);
        }

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}