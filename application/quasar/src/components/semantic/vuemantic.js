import jQuery from 'jquery';


function installed(Vue)	{
    var semantic = require('./semantic.min');

    Vue.directive('ui-transition', require('./semantic-transition.directive').default);
    Vue.directive('ui-accordion', require('./semantic-accordion.directive').default);
    Vue.directive('ui-dimmer', require('./semantic-dimmer.directive').default);
    Vue.directive('ui-dropdown', require('./semantic-dropdown.directive').default);
    Vue.directive('ui-modal', require('./semantic-modal.directive').default);
    Vue.directive('ui-search', require('./semantic-search.directive').default);
    Vue.directive('ui-sidebar', require('./semantic-sidebar.directive').default);
    Vue.directive('ui-progress', require('./semantic-progress.directive').default);

	Object.defineProperties(Vue.prototype, {

	});
};

export default installed;
