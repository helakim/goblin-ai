import $ from 'jquery';


class Progress    {
    constructor(el, options = {
        /**
         * Whether success state should automatically trigger when progress completes
         */
        autoSuccess: true,

        /**
         * Whether progress should automatically show activity when incremented
         */
        showActivity: true,

        /**
         * When set to true, values that calculate to above 100% or below 0% will be adjusted. When set to false, inappropriate values will produce an error.
         */
        limitValues: true,

        /**
         * Can be set to either to display progress as percent or ratio. Matches up to corresponding text template with the same name.
         */
        label: 'percent',

        /**
         * When incrementing without value, sets range for random increment value
         */
        random: {
            active  : 'active',
            error   : 'error',
            success : 'success',
            warning : 'warning'
        },

        /**
         * Decimal point precision for calculated progress
         */
        precision: 1,

        /**
         * Setting a total value will make each call to increment get closer to this total (i.e. 1/20, 2/20 etc)
         */
        total: false,

        /**
         * Sets current value, when total is specified, this is used to calculate a ratio of the total, with percent this should be the overall percent
         */
        value: false,

        /**
         * Callback on percentage change
         */
        onChange: function(percent, value, total)   {},

        /**
         * Callback on success state
         */
        onSuccess: function(total)  {},

        /**
         * Callback on active state
         */
        onActive: function(value, total)    {},

        /**
         * Callback on error state
         */
        onError: function(value, total) {},

        /**
         * Callback on warning state
         */
        onWarning: function(value, total)   {},

        /**
         * Class names used to attach style to state
         */
        className: {
            active  : 'active',
            error   : 'error',
            success : 'success',
            warning : 'warning'
        },
    })   {
        this.$el = $(el).progress(options);
    };

    /**
     * Sets current percent of progress to value. If using a total will convert from percent to estimated value.
     */
    setPercent(percent) { this._behavior('set percent', percent); };

    /**
     * Sets progress to specified value. Will automatically calculate percent from total.
     */
    setProgress(value) { this._behavior('set progress', value); };

    /**
     * Increments progress by increment value, if not passed a value will use random amount specified in settings
     */
    increment(incrementValue) { this._behavior('increment', incrementValue); };

    /**
     * Decrements progress by decrement value, if not passed a value will use random amount specified in settings
     */
    decrement(decrementValue) { this._behavior('decrement', decrementValue); };

    /**
     * Immediately updates progress to value, ignoring progress animation interval delays
     */
    updateProgress(value) { this._behavior('update progress', value); };

    /**
     * Finishes progress and sets loaded to 100%
     */
    complete() { this._behavior('complete'); };

    /**
     * Resets progress to zero
     */
    reset() { this._behavior('reset'); };

    /**
     * Set total to a new value
     */
    setTotal(total) { this._behavior('set total', total); };

    /**
     * Replaces templated string with value, total, percent left and percent.
     */
    getText(text) { return this._behavior('get text', text); };

    /**
     * Returns normalized value inside acceptable range specified by total.
     */
    getNormalizedValue(value) { return this._behavior('get normalized value', value); };

    /**
     * Returns percent as last specified
     */
    getPercent() { return this._behavior('get percent'); };

    /**
     * Returns current progress value
     */
    getValue() { return this._behavior('get value'); };

    /**
     * Returns total
     */
    getTotal() { return this._behavior('get total'); };

    /**
     * Returns whether progress is completed
     */
    isComplete() { return this._behavior('is complete'); };

    /**
     * Returns whether progress was a success
     */
    isSuccess() { return this._behavior('is success'); };

    /**
     * Returns whether progress is in warning state
     */
    isWarning() { return this._behavior('is warning'); };

    /**
     * Returns whether progress is in error state
     */
    isError() { return this._behavior('is error'); };

    /**
     * Returns whether progress is in active state
     */
    isActive() { return this._behavior('is active'); };

    /**
     * Sets progress to active state
     */
    setActive() { this._behavior('set active'); };

    /**
     * Sets progress to warning state
     */
    setWarning() { this._behavior('set warning'); };

    /**
     * Sets progress to success state
     */
    setSuccess() { this._behavior('set success'); };

    /**
     * Sets progress to error state
     */
    setError() { this._behavior('set error'); };

    /**
     * Changes progress animation speed
     */
    setDuration(value) { this._behavior('set duration(value)'); };

    /**
     * Sets progress exterior label to text
     */
    setLabel(text) { this._behavior('set label(text)'); };

    /**
     * Sets progress bar label to text
     */
    setBarLabel(text) { this._behavior('set bar label(text)'); };

    /**
     * Removes progress to active state
     */
    removeActive() { this._behavior('remove active'); };

    /**
     * Removes progress to warning state
     */
    removeWarning() { this._behavior('remove warning'); };

    /**
     * Removes progress to success state
     */
    removeSuccess() { this._behavior('remove success'); };

    /**
     * Removes progress to error state
     */
    removeError() { this._behavior('remove error'); };


    _behavior(...args) {
        return $.fn.progress.apply(this.$el, args);
    };
};


export default {
    bind: function(el, binding) {

    },

    inserted: function(el, binding, vnode)    {
        var progress = new Progress(el, binding.value.setting);

        vnode.context[binding.value.el] = progress;
    },

    update: function() {

    },

    componentUpdated: function() {

    },

    unbind: function(el) {

    },
};
