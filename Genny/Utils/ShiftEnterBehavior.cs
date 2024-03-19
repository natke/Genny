using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace Genny.Utils
{
    /// <summary>
    /// Behaviour to use Shift + Enfer to add a new line to a TextBox allowing IsDefault Commands to be fired on Enter
    /// </summary>
    public class ShiftEnterBehavior
    {
        public static readonly DependencyProperty EnableProperty =
            DependencyProperty.RegisterAttached("Enable", typeof(bool), typeof(ShiftEnterBehavior), new PropertyMetadata(false, OnEnableChanged));

        public static bool GetEnable(DependencyObject obj)
        {
            return (bool)obj.GetValue(EnableProperty);
        }

        public static void SetEnable(DependencyObject obj, bool value)
        {
            obj.SetValue(EnableProperty, value);
        }

        private static void OnEnableChanged(DependencyObject obj, DependencyPropertyChangedEventArgs e)
        {
            if (obj is TextBox textBox)
            {
                bool attach = (bool)e.NewValue;

                if (attach)
                {
                    textBox.PreviewKeyDown += TextBox_PreviewKeyDown;
                }
                else
                {
                    textBox.PreviewKeyDown -= TextBox_PreviewKeyDown;
                }
            }
        }

        private static void TextBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter && Keyboard.Modifiers == ModifierKeys.Shift)
            {
                if (sender is TextBox textBox)
                {
                    e.Handled = true;
                    textBox.AppendText(Environment.NewLine);
                    textBox.CaretIndex = textBox.Text.Length;
                }
            }
        }
    }
}
